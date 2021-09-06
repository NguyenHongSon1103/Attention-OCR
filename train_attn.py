import sys
sys.path.append('modules')
import tensorflow as tf
from dataset import Dataset
from model_attention import Model
import numpy as np
import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# ** Define parameter **#
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', required=True, type=str, help='path to image source')
parser.add_argument('--train_label_path', required=True, type=str, help='path to label')
parser.add_argument('--valid_data_path', required=True, type=str, help='path to image source')
parser.add_argument('--valid_label_path', required=True, type=str, help='path to label')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--charset_path', required=True, type=str, help='characters set')
parser.add_argument('--max_length', default=12, type=int, help='max sequence length')
parser.add_argument('--input_h', default=64, type=int, help='input height')
parser.add_argument('--input_w', default=416, type=int, help='input width')

args = parser.parse_args()

# ** Prepare data ** #
print('---------- loading data ! ----------')
trainingset = Dataset(args.train_data_path, args.train_label_path, args.charset_path, batch_size=args.batch_size,
                input_size=(args.input_w, args.input_h), max_sequence_length=args.max_length)
print('Found %d images in training set'%(len(trainingset)*args.batch_size))
validset = Dataset(args.valid_data_path, args.valid_label_path, args.charset_path, batch_size=args.batch_size,
                    input_size=(args.input_w, args.input_h), max_sequence_length=args.max_length)
print('Found %d images in validation set'%(len(validset)*args.batch_size))

# ** Setup params of model ** #
cutted_layer_name = 'block6a_activation'
bilstm_units = 128
lstm_units = 256
checkpoint_path = 'checkpoint/%s_%d_%d'%(cutted_layer_name, bilstm_units, lstm_units)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
# ** Build model ** #
if not os.path.exists(checkpoint_path + '/' + 'best_loss.h5'):
    model = Model(input_shape=(args.input_h, args.input_w, 3), max_seq_len=args.max_length, chars_size=trainingset.chars_size,
                 feature_extractor='Efficient', bilstm_units=bilstm_units, lstm_units=lstm_units).make_model(training=True)
else:
    with open(checkpoint_path + '/' + 'model_arc.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(checkpoint_path + '/' + 'best_loss.h5')
print("Loading model done !")
# model.summary()

# ** Define optimizer , loss and metric ** #
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
loss_obj = tf.keras.losses.categorical_crossentropy
def loss_function(y_true, y_pred):
    """
        Args:
            y_true and y_pred with shape (bs, max_seq_len, chars_size)
        Output: 
            crossentropy of y_true and y_pred
    # """
    loss = loss_obj(y_true, y_pred)
    # loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = tf.reduce_mean(loss)
    return loss

acc_metric = tf.keras.metrics.CategoricalAccuracy(name='acc')
    
# ** Callbacks ** #
with open(checkpoint_path + '/' + 'model_arc.json', 'w') as fj:
    fj.write(model.to_json())

checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(checkpoint_path + '/' + 'best_loss.h5', save_weights_only=True, mode='min',
                                                save_best_only=True)
checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(checkpoint_path + '/' + 'best_acc.h5', save_weights_only=True, mode='max', monitor='val_acc',                                                                                                                                                save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(update_freq='batch')

model.compile(optimizer=optimizer, loss=loss_function, metrics=[acc_metric])
model.fit(trainingset, batch_size=args.batch_size, epochs=30, callbacks=[checkpoint_loss, checkpoint_acc, tensorboard],
        validation_data=validset)