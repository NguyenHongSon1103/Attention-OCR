from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Concatenate

class BidirectionalLSTM:
    def __init__(self, input_size=128):
        # self.dense = Dense(input_size)
        self.bilstm = Bidirectional(LSTM(input_size, return_sequences=True, return_state=True))

    def __call__(self, feature_map):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x 2*output_size] 
        """
        # bilstm_out, bilstm_state = self.bilstm(feature_map)
        # return bilstm_out, bilstm_state
        # feature_map = self.dense(feature_map)
        out = self.bilstm(feature_map)
        bilstm_out, hs_fw , cs_fw, hs_bw, cs_bw = out
        last_hidden_state = Concatenate()([hs_fw, hs_bw])
        last_cell_state = Concatenate()([cs_fw, cs_bw])
        return bilstm_out, last_hidden_state, last_cell_state

if __name__ == '__main__':
    # Use for test shape
    import numpy as np
    batch_size = 10
    seq_len = 26
    depth = 1280
    lstm_units = 128
    tensor = np.random.random((batch_size, seq_len, depth))
    bilstm = BidirectionalLSTM(input_size=lstm_units)
    bilstm_out, last_hidden_state, last_cell_state = bilstm(tensor)
    print(bilstm_out)
    """
    From (10, 26, 1280) -> (10, 26, 256)
    <==> (batch_size, seq_len, depth) -> (batch_size, seq_len, lstm_units*2)
    """