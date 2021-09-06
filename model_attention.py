import tensorflow as tf
import sys
sys.path.append('modules')
from feature_extractor import EfficientNetB0
from sequence_modeling import BidirectionalLSTM
from attention import Attention

class Model:
    def __init__(self, input_shape=(64, 416, 3), cutted_layer_name='block6a_activation',
                bilstm_units=256, lstm_units=128, chars_size=11, max_seq_len=12):
        # ** Feature Extractor ** #
        self.feature_extractor = EfficientNetB0(input_shape=input_shape, cutted_layer_name=cutted_layer_name)
        # ** Sequence Modeling ** #
        self.sequence_model = BidirectionalLSTM(input_size=bilstm_units)
        self.decoder = Attention(lstm_units, chars_size, max_seq_len)
        self.chars_size = chars_size
        self.max_seq_len = max_seq_len

    def make_model(self, training=True, export=False):
        """
        export: True when make saved model for serving
        """
        input_tensor = self.feature_extractor.model.input
        input_init_vec = tf.keras.layers.Input(shape=tf.TensorShape((1, self.chars_size)))
        if export:
            normarlize = input_tensor / 255.0
            feature_map = self.feature_extractor(normarlize)
        else:
            feature_map = self.feature_extractor(input_tensor)
        # print('Feature map shape: ',feature_map.shape) # Efficient: Nonex26x1280
        state_h = None
        if self.sequence_model is not None:
            feature_map, state_h, _ = self.sequence_model(feature_map) 
        if training: 
            input_target_label = tf.keras.layers.Input(shape=tf.TensorShape((self.max_seq_len, self.chars_size)))
            dec_output = self.decoder(input_init_vec, state_h, feature_map, training, input_target_label)
            model = tf.keras.models.Model(inputs=[input_tensor, input_init_vec, input_target_label], outputs=dec_output)
        else:
            if export:
                input_init_vec = tf.zeros((1, 1, 11))             
                dec_output = self.decoder(input_init_vec, state_h, feature_map)
                model = tf.keras.models.Model(inputs=input_tensor, outputs=dec_output)
            else:
                dec_output = self.decoder(input_init_vec, state_h, feature_map)
                model = tf.keras.models.Model(inputs=[input_tensor, input_init_vec], outputs=dec_output)
        
        return model

if __name__ == '__main__':
    model = Model(input_shape=(64, 416, 3), feature_extractor='Efficient',bilstm_units=256,
                    lstm_units=256, chars_size=11, max_seq_len=12).make_model(training=False)
    model.summary()
    '''
    infer model
    Total params: 5,114,926
    Trainable params: 5,072,903
    Non-trainable params: 42,023    

    train model
    Total params: 5,114,926
    Trainable params: 5,072,903
    Non-trainable params: 42,023    
    '''