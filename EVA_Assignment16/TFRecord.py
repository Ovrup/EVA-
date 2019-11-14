import numpy as np
import tensorflow as tf
import io


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 



class convert_tfrecord:
  def create_tfrecord(self,features,labels,output_dir):
    with tf.python_io.TFRecordWriter(output_dir) as file:
      for i,each_feature in (enumerate(features)):
        data = each_feature
        label = np.argmax(labels[i])
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data.tobytes()),
                'label': _int64_feature(label)
            }))
        file.write(example.SerializeToString())