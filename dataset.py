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

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data_source_path, label_path, charset_path, batch_size=32, input_size=(416, 46), max_sequence_length=12):
        with open(label_path, 'r') as f:
            self.labels = f.read().split('\n')
        self.batch_size = batch_size
        self.data_source_path = data_source_path
        self.resize_w, self.resize_h = input_size
        self.max_sequence_length = max_sequence_length
        self.c2i, self.i2c = mapper(charset_path)
        self.chars_size = len(self.i2c)

    def seq2onehot(self, sequence, max_sequence_length=12):
        onehot_tensor = np.zeros((max_sequence_length, len(self.i2c))) #create matrix (12, 11)
        for i, c in enumerate(sequence):
            onehot_tensor[i][self.c2i[c]] = 1
        for i in range(len(sequence), max_sequence_length):
            onehot_tensor[i][-1] = 1
        return onehot_tensor

    def on_epoch_end(self):
        np.random.shuffle(self.labels)
    
    def __len__(self):
        return len(self.labels) // self.batch_size
    
    def __getitem__(self, idx):
        raws = self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        raws = [label.split('\t') for label in raws]
        filenames = [label[0] for label in raws]
        labels = [label[1] for label in raws]
        images = []
        # ** Prepare for images ** #
        for filename in filenames:
            fp = os.path.join(self.data_source_path, filename)
            img = cv2.imread(fp)
            h, w = img.shape[:2]
            # lazy resize
            # img = cv2.resize(img, self.input_size) / 255.0
            # resize with padding
            w_new = int(w/h * self.resize_h)
            
            img = cv2.resize(img, (w_new, self.resize_h))
            if w_new < self.resize_w:
                ps = self.resize_w - w_new
                img = cv2.copyMakeBorder(img, 0, 0, 0, ps, cv2.BORDER_CONSTANT, (0, 0, 0))
            else:
                img = cv2.resize(img, (self.resize_w, self.resize_h))
            img = img / 255
            images.append(img)
        
        # ** Prepare for labels ** #
        onehot_labels = []
        for label in labels:
            onehot = self.seq2onehot(label, max_sequence_length=self.max_sequence_length)
            onehot_labels.append(onehot)
        onehot_labels = np.array(onehot_labels)
        images = np.array(images)
        return ([images, np.zeros((self.batch_size, 1, self.chars_size)), onehot_labels], onehot_labels)
            
        


