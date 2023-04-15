import re
import numpy as np
import tensorflow as tf


IMAGE_SIZE = None
IMAGE_COUNT = 0

def set_image_size(image_size):
    global IMAGE_SIZE
    IMAGE_SIZE = image_size


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    count = 0
    for fn in filenames:
        for _ in tf.compat.v1.python_io.tf_record_iterator(fn):
            count += 1
    return count


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image, seed=42)
    image = tf.image.random_flip_up_down(image, seed=42)
    image = tf.image.random_saturation(image, 0, 2, seed=42)
    image = tf.image.random_brightness(image, .1, seed=42)
    image = tf.image.random_hue(image, .1, seed=42)
    image = tf.image.random_jpeg_quality(image, 0, 2, seed=42)
    return image, label


def decode_image(image_data):
    global IMAGE_SIZE, IMAGE_COUNT
    if IMAGE_SIZE is None:
        raise ValueError("Please set the IMAGE_SIZE")
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    IMAGE_COUNT = image.shape[0]
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    return image # returns a dataset of images


def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = ordered # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE) # automatically interleaves reads from multiple files 
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_dataset(filenames, labeled=True, ordered=False, repeat=1, augment=False, shuffle=False, batch_size=32, buffer_size=2048):
    dataset = load_dataset(filenames, labeled, ordered)
    if augment:
        dataset = dataset.map(data_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(repeat) # the training dataset must repeat for several epochs
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
