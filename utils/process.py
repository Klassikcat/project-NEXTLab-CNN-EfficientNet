import math, re, os
import tensorflow as tf
import numpy as np
from utils.tfrecordViewer import TFRecordLoader

class ReadTFRecord():
    def __init__(self, tfrecord_path, label, batch_size=int, image_size=int, channels=3):
        assert type(label) == str
        self.loader = TFRecordLoader(tfrecord_path=tfrecord_path)
        self.data = self.loader.load()
        self.features = self.loader.feature
        self.batch_size = batch_size
        self.image_size = [image_size, image_size, channels]
        self.label = label
        self.AUTO = tf.data.experimental.AUTOTUNE

    def decode_img(self, file):
        image = tf.io.decode_jpeg(file)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.image_size[:2])
        image = tf.reshape(image, self.image_size)
        return image

    def _decode_dataset(self, features):
        if self.label == 'brand':
            path  = features['image/path']
            brand = features['image/class/brand']
            brand = tf.compat.as_str_any(brand)
            image = features['image/encoded']
            image = self.decode_img(image)
            return image, brand
        else:
            path  = features['image/path']
            brand = features['image/class/brand']
            brand = tf.compat.as_str_any(brand)
            color = features['image/class/color']
            color = tf.compat.as_str_any(color)
            model = features['image/class/model']
            model = tf.compat.as_str_any(model)
            year  = features['image/class/year']
            year = tf.compat.as_str_any(year)
            image = features['image/encoded']
            image = self.decode_img(image)
            return image, brand, color, model, year

    def _fliping_augmentation(self, label, image):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def load_dataset(self, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False  # disable order, increase speed

        dataset = self.data.with_options(ignore_order)
        dataset = dataset.map(self._decode_dataset)
        return dataset

    def call(self):
        dataset = self.load_dataset()
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def call_vali(self):
        dataset = self.load_dataset()
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO)
        return dataset

    def call_test(self):
        dataset = self.load_dataset()
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO)
        return dataset


def main():
   tfrecord = ReadTFRecord(tfrecord_path='/Volumes/External SSD for Data/NEXTLab/contents/train/tfrecord/001-35459.tfrecord', label='all', batch_size=32, image_size=224)
   record = tfrecord.call()
   print("Done!")

if __name__ == '__main__':
    main()