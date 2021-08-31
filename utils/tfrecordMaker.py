import os
import glob
import numpy as np
import json
import argparse
import math
from tqdm.notebook import tqdm

import pandas as pd
import tensorflow as tf

_DEFAULT_DATA_PATH = 'data'
_DEFAULT_LABEL_PATH = 'label'
_DEFAULT_OUTPUT_DIR = 'tfrecord'

_DEFAULT_N_SHARDS = 1

_SEED = 42

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_example(path, img_str, brand_name, color, model, year):
    """
    :param img_str: byte-encoded image, str
    :param brand_name: brand name of car, str
    :param color: car's color name, str
    :param model: car's model, str
    :param year: car's year, str
    :return: tf.train.Example file
    """
    feature = {'image/path': _bytes_feature(path),
               'image/encoded': _bytes_feature(img_str),
               'image/class/brand': _bytes_feature(brand_name),
               'image/class/color': _bytes_feature(color),
               'image/class/model': _bytes_feature(model),
               'image/class/year': _bytes_feature(year)
               }
    return tf.train.Example(features=tf.train.Features(feature=feature))

class TFRecordConverter:
    def __init__(self, origin_path, data_path, label_path, output_dir, n_shards):
        """
        :param origin_path: base path that contains data/label. TYPE:string
        :param data_path: sub-path that contains train data. DEFAULT: data(str)
        :param label_path: sub-path that contains train label. DEFAULT: label(str)
        """
        self.origin_path = origin_path
        self.data_path = data_path
        self.label_path = label_path
        self.output_dir = output_dir
        self.n_shards = n_shards

    __SUBDIR_JSON = '/**/**/**.json'

    def get_paths(self):
        paths = glob.glob(self.origin_path + self.label_path + self.__SUBDIR_JSON)
        return paths

    def get_jsons(self, paths=list):
        self.paths = paths
        json_dataframe = pd.DataFrame()
        for path in tqdm(paths):
            with open(path) as json_file:
                json_data = json.load(json_file)
                json_data = json_data['car']
                data = json_data['attributes']
                data['image_path'] = self.origin_path + self.data_path + '/' + json_data['imagePath']
                json_dataframe = json_dataframe.append(data, ignore_index=True)
        return json_dataframe

    def get_img(self, image_path=str):
        self.image_path = image_path
        files = tf.io.read_file(image_path)
        return tf.image.decode_image(files, channels=3)

    def _get_shard_path(self, shard_id, shard_size):
        return os.path.join(self.origin_path,
                            self.output_dir,
                            f'{shard_id:03d}-{shard_size}.tfrecord')

    def _write_tfrecord_file(self, df, indices, shard_path):
        self.df = df
        self.shard_path = shard_path
        self.indices = indices
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(shard_path, options=options) as out:
            for index in indices:
                file_path = df.image_path.iloc[index].encode()
                brand = df.brand.iloc[index].encode()
                color = df.color.iloc[index].encode()
                model = df.model.iloc[index].encode()
                year = df.year.iloc[index].encode()
                decoded_image = open(df.image_path.iloc[index], 'rb').read()
                example = make_example(file_path, decoded_image, brand, color, model, year)
                out.write(example.SerializeToString())

    def convert(self):
        paths = self.get_paths()
        dataframe = self.get_jsons(paths)
        size = len(dataframe)
        offset = 0
        shard_size = math.ceil(size/self.n_shards)
        cumulative_size = offset + size
        for shard_id in range(1, self.n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(shard_id, step_size)
            file_indices = np.arange(offset, offset + step_size)
            self._write_tfrecord_file(dataframe, file_indices, shard_path)
            offset += step_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-path', type=str, dest='origin_path',
                        help='Absolute path of the target path.'
                             '(example = /User/administrator/project-path/contents/)')
    parser.add_argument('--data-path', type=str, dest='data_path',
                        default=_DEFAULT_DATA_PATH,
                        help='relative path of the data path in the project file.'
                            f'(default: {_DEFAULT_DATA_PATH})')
    parser.add_argument('--label-path', type=str, dest='label_path',
                        default=_DEFAULT_LABEL_PATH,
                        help='relative path of the label path in the project file.'
                            f'(defaults: {_DEFAULT_LABEL_PATH})')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='relative directory in the project'
                             'that tfrecord file will be saved.'
                            f'(defaults:{_DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--n-shards', type=int, dest='n_shards', default=1,
                        help='number of shards to divide dataset TFRecord into.'
                             '(defaults: 1)')
    return parser.parse_args()

def main(args):
    converter = TFRecordConverter(args.origin_path,
                                  args.data_path,
                                  args.label_path,
                                  args.output_dir,
                                  args.n_shards)
    converter.convert()

if __name__ == '__main__':
    main(parse_args())