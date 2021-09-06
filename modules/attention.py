import tensorflow as tf
import numpy as np


def attention_step(query, value, key):
    """
        Attention with Luong's Style
        Args:
            query: A tensor of shape (batch_size, Tq, dim) 
            value: A tensor of shape (batch_size, Tv, dim)
            key: Usually as same as value
        Note: Remember make query's dim and value's dim equal before fit in to attention
        Output shape: Context tensor of shape (batch_size, Tq, dim_q) , Attention weights with shape (batch_size, Tq, Tv)
    """
    # Calculate score_qk shape (batch_size, Tq, Tv)
    score_qk = tf.matmul(query, key, transpose_b=True)
    distribution = tf.nn.softmax(score_qk) 
    
    # Get context vector (batch_size, Tq, Tv) x (batch_size, Tv, dim) = (batch_size, Tq, dim): 
    context_vec = tf.matmul(distribution, value)

    return context_vec, distribution

class Attention:
    def __init__(self, lstm_units, chars_size, max_seq_len):
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_state=True)
        self.wq = tf.keras.layers.Dense(lstm_units)
        self.wk = tf.keras.layers.Dense(lstm_units)
        self.wv = tf.keras.layers.Dense(lstm_units)

        self.fc = tf.keras.layers.Dense(chars_size, activation=tf.nn.softmax)
        self.attn = attention_step
        self.max_seq_len = max_seq_len
        self.chars_size = chars_size
        # self.attn2 = tf.keras.layers.Attention()
    
    def __decode_step__(self, context_t, input_t):

        """
            Args:
                context_t: context vector from attention at time step t (bs, 1, dim)
                input_t : input at time step t (bs, 1, chars_size)
            Output: 
                output_t: output at time step t (bs, 1, chars_size)
                hidden_t: hidden state at time step t (bs, 1, dim)
        """
        attention_t = tf.keras.layers.Concatenate()([context_t, input_t])
        lstm_out_t = self.lstm(attention_t)
        output_t, hidden_t = lstm_out_t[1], lstm_out_t[2]

        hidden_t = tf.expand_dims(hidden_t, 1)
        output_t = self.fc(output_t)
        output_t = tf.expand_dims(output_t, 1)

        return hidden_t, output_t

    def __to_one_hot__(self, distributed_input):
        """
        Args:
            distributed_input: A probality tensor of shape (bs, 1, chars_size) 
        """
        distributed_input_sq = tf.squeeze(distributed_input, 1) #convert to (bs, chars_size)
        # print(distributed_input_sq)
        idxs = tf.argmax(distributed_input_sq, -1) # (bs)
        # print(idxs)
        onehot_output = tf.one_hot(idxs, self.chars_size)
        onehot_output = tf.expand_dims(onehot_output, 1)
        # print(onehot_output)
        return onehot_output

    def __call__(self, init_input, enc_cell_state, encoder_hidden_states, training=False, target_label=None):
        """
            Input:  
                init_input: zero's tensor with shape (bs, 1, chars_size)
                encoder cell state - last cell state of encoder with shape (bs, depth_enc)
                encoder_hidden_states - sequences of encoder hidden state with shape (bs, T, depth_dec)
                training: determine if train or inference
                target_label: Onehot vecto with shape (bs, seq_len, chars_size)
            Output: 
                A distributed tensor of shape (bs, seq_len, chars_size)
        """
        # ** Convert query and key to (bs, 1, dim) and (bs, T, dim)
        enc_cell_state = self.wq(enc_cell_state) 
        ehs_key = self.wk(encoder_hidden_states)
        ehs_value = self.wv(encoder_hidden_states)
        
        # ** Prepare initial input and decoder hidden_state ** #
        input_t = init_input
        context_vec = tf.expand_dims(enc_cell_state, 1)
        # ** prepare output ** #
        dec_output = tf.zeros_like(init_input)
        # ** Training mode: Decoder input in each step is label's character onehot at that index ** #
        if training:
            for t in range(self.max_seq_len):           
                current_hidden_state, input_t = self.__decode_step__(context_vec, input_t)
                context_vec, attn_weight = self.attn(current_hidden_state, ehs_key, ehs_value)
                # context_vec = self.attn2([current_hidden_state, encoder_hidden_states])
                dec_output = tf.concat([dec_output, input_t], 1)
                input_t = tf.expand_dims(target_label[:, t], 1)
            return dec_output[:, 1:]
        else:
            for t in range(self.max_seq_len):           
                current_hidden_state, input_t = self.__decode_step__(context_vec, input_t)
                context_vec, attn_weight = self.attn(current_hidden_state, ehs_key, ehs_value)
                # context_vec = self.attn2([current_hidden_state, encoder_hidden_states])
                dec_output = tf.concat([dec_output, input_t], 1)
                input_t = self.__to_one_hot__(input_t)
            return dec_output[:, 1:]

