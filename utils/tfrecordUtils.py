import tensorflow as tf
import matplotlib.pyplot as plt
import platform
import os
import glob
import pandas as pd
import json
from tqdm.auto import tqdm

class TFRecordLoader:
    def __init__(self, origin_path = '',tfrecord_path = ''):
        self.path = os.path.join(origin_path,tfrecord_path)
        
    def get_tfrecord(self, path) : 
        return tf.data.TFRecordDataset(path, compression_type = 'GZIP')
    
    def _parse_image_function(self, example_proto):
        feature = {
            'image/path': tf.io.FixedLenFeature((), tf.string),
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/class/brand': tf.io.FixedLenFeature((), tf.string),
            'image/class/color': tf.io.FixedLenFeature((), tf.string),
            'image/class/model': tf.io.FixedLenFeature((), tf.string),
            'image/class/year': tf.io.FixedLenFeature((), tf.string)
        }
        return tf.io.parse_single_example(example_proto, feature)

    def load(self):
        tfrecord = self.get_tfrecord(self.path)
        return tfrecord.map(self._parse_image_function)

class TFRecordViewer:
    def __init__(self, n_image = 1):
        self.n_image = n_image

        if platform.system() == 'Darwin': #Mac
            plt.rc('font', family='AppleGothic') 
        elif platform.system() == 'Windows': #Window
            plt.rc('font', family='Malgun Gothic') 
        elif platform.system() == 'Linux': #Linux or Colab
            plt.rc('font', family='Malgun Gothic') 
        plt.rcParams['axes.unicode_minus'] = False #resolve minus symbol breaks when using Hangul font
    
    def _tensor_decode(self, features):
        path  = features['image/path'].numpy().decode()
        brand = features['image/class/brand'].numpy().decode()
        color = features['image/class/color'].numpy().decode()
        model = features['image/class/brand'].numpy().decode()
        year  = features['image/class/model'].numpy().decode()
        image = features['image/encoded']
        image = tf.io.decode_jpeg(image)
        image = tf.keras.preprocessing.image.array_to_img(image)

        return path, brand, color, color, model, year, image

    def show(self, parsed_tfrecord):
        self.parsed_tfrecord = parsed_tfrecord

        for features in self.parsed_tfrecord:          
            path, brand, color, color, model, year, image = \
            self._tensor_decode(features)          
            print(path)
            print(brand)
            print(color)
            print(model)
            print(year)
            plt.text(180, 0, 'path : '+ path)
            plt.text(180, 10, 'brand : '+ brand)
            plt.text(180, 20, 'color : '+ color)
            plt.text(180, 30, 'model : '+ model)
            plt.text(180, 40, 'year : '+ year)

            # plt.text(10.0, 1, 'brand : '+ brand)
            
            plt.imshow(image)
            plt.show()

class TFRecordChecker:
    def __init__(self, tfrecord_path):
        self.path = tfrecord_path
    
    def get_length(self, file_name):
        tmp = file_name.split('-')
        tmp = tmp[1].split('.')
        return int(tmp[0])

    def check_tfrecord(self):
        total_images = 0
        tfrecord_pathes = sorted(glob.glob(os.path.join(self.path, '*')))
        print("=========================================================================================================================")
        print("#                                                      Checking...                                                      #")
        print("=========================================================================================================================") 
           
        for idx, tfrecord_path in enumerate(tfrecord_pathes):
            fname, ext = os.path.splitext(tfrecord_path)
            assert ext == '.tfrecord', 'it is not a tfrecord file.'
            try:
                total_images += sum([1 for _ in tqdm(tf.data.TFRecordDataset(tfrecord_path, compression_type = 'GZIP'),
                                                                             total = self.get_length(tfrecord_path))]) # Check corrupted tf records
                print("{}: {} is ok".format(idx, tfrecord_path))
            except:
                print("{}: {} is corrupted".format(idx, tfrecord_path))

        print("Succeed, no corrupted tf records found for {} images".format(total_images))

