""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
"""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="TensorFlow TFRecord converter")
parser.add_argument("-l",
                    "--label_path",
                    help="Path to the labels txt file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str)

args = parser.parse_args()

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_labels(lb_path):
    with open(lb_path, 'r') as f:
        lines = f.read().split('\n')
    
    lines = [line.split('\t') for line in lines if line != '']
    labels = {line[0]:line[1] for line in lines}
    return labels

def create_tf_example(img_name, path, label):
    img_string = open(os.path.join(path, img_name), 'rb').read()
    filename = img_name.encode('utf8')
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(filename),
        'encoded': _bytes_feature(img_string),
        'label': _bytes_feature(label.encode('utf8')),
    }))
    return tf_example


def main():
    count_success, count_fail = 0, 0
    writer = tf.io.TFRecordWriter(args.output_path)
    labels = get_labels(args.label_path)
    
    for img_name in tqdm(labels.keys()):
        fp = os.path.join(args.image_dir, img_name)
        if not os.path.exists(fp):
            count_fail += 1
            continue
        tf_example = create_tf_example(img_name, args.image_dir, labels[img_name])
        writer.write(tf_example.SerializeToString())
        count_success += 1

    writer.close()
    print('Filter %d samples have label but without image'%count_fail)
    print('Successfully created the TFRecord file %s with %d samples'%(args.output_path, count_success))

if __name__ == '__main__':
    main()