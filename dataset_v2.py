import tensorflow as tf
import numpy as np
import cv2
import os

def mapper(charset_path):
    """
    save character set with format: index   char
    """
    with open(charset_path, 'r') as f:
        lines = f.read().split('\n')
    
    lines = [line.split('\t') for line in lines if line != '']
    c2i = {line[1]:int(line[0]) for line in lines}
    i2c = [line[1] for line in lines]

    return c2i, i2c

def parse_tf_records(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    image_feature_description = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'encoded': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_dataset.map(_parse_image_function)
    return parsed_image_dataset

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, tf_record_path, charset_path, batch_size, input_size, max_sequence_length):
        self.batch_size = batch_size
        self.resize_w, self.resize_h = input_size
        self.max_sequence_length = max_sequence_length
        self.c2i, self.i2c = mapper(charset_path)
        self.chars_size = len(self.i2c)
        self.data = list(parse_tf_records(tf_record_path))
        
    def seq2onehot(self, sequence, max_sequence_length=12):
        onehot_tensor = np.zeros((max_sequence_length, len(self.i2c))) #create matrix (12, 11)
        for i, c in enumerate(sequence):
            onehot_tensor[i][self.c2i[c]] = 1
        for i in range(len(sequence), max_sequence_length):
            onehot_tensor[i][-1] = 1
        return onehot_tensor
    
    def __preprocess__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        w_new = int(w/h * self.resize_h) 
        img = cv2.resize(img, (w_new, self.resize_h))
        if w_new < self.resize_w:
            ps = self.resize_w - w_new
            img = cv2.copyMakeBorder(img, 0, 0, 0, ps, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            img = cv2.resize(img, (self.resize_w, self.resize_h))
        img = img / 255.0
        return img

    def on_epoch_end(self):
        np.random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data) // self.batch_size 

    def __getitem__(self, idx):
        start = self.batch_size*idx
        end = start + self.batch_size
        batches = self.data[start:end]
        images, onehot_labels = [], []
        for sample in batches:
            img = sample['encoded'].numpy()
            label = sample['label'].numpy()

            img = np.fromstring(img, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            print(img.shape)
            images.append(self.__preprocess__(img))

            label = label.decode()
            onehot = self.seq2onehot(label, max_sequence_length=self.max_sequence_length)
            onehot_labels.append(onehot)

        onehot_labels = np.array(onehot_labels)
        images = np.array(images)
        return ([images, np.zeros((self.batch_size, 1, self.chars_size)), onehot_labels], onehot_labels)
            
        
if __name__ == '__main__':
    import pdb
    tf_record_path = 'test.record'
    charset_path = r'charset\number.txt'
    data = Dataset(tf_record_path, charset_path, 3, (416, 64), 13)
    print(data[0])
    # pdb.set_trace()

