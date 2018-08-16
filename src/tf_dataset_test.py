#-*- coding: UTF-8 -*-  
import tensorflow as tf
import numpy as np
import ipdb; ipdb.set_trace()


# dataset = tf.contrib.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
dataset = tf.contrib.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                       
        "b": np.random.uniform(size=(5, 2))
    }
)


iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# 图片文件的列表
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg"])
# label[i]就是图片filenames[i]的label
labels = tf.constant([0, 37])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)

iterator = dataset.make_one_shot_iterator()
one_element = get_next()