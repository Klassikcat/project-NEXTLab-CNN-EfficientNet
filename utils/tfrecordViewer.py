import argparse
import platform

import matplotlib.pyplot as plt
import tensorflow as tf

_DEFAULT_TFRECORD_DIR = 'tfrecord' 
# : Example DIR = 'data/tfrecord/001-744.tfrecord'

class TFRecordLoader:
    def __init__(self, tfrecord_path):
        """[summary]

        Args:
            tfrecord_path ([str of path or list of paths]): []
        """
        self.tfrecord_path = tfrecord_path

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
        tfrecord = self.get_tfrecord(self.tfrecord_path)
        return tfrecord.map(self._parse_image_function)

class TFRecordViewer(TFRecordLoader):
    def __init__(self, n_image=int):
        self.n_image = n_image

        if platform.system() == 'Darwin': #Mac
            plt.rc('font', family='AppleGothic') 
        elif platform.system() == 'Windows': #Window
            plt.rc('font', family='Malgun Gothic') 
        elif platform.system() == 'Linux': #Linux or Colab
            plt.rc('font', family='NanumBarunGothic')
        plt.rcParams['axes.unicode_minus'] = False #resolve minus symbol breaks when using Hangul font
    
    def _tensor_decode(self, features):
        path  = features['image/path'].numpy().decode()
        brand = features['image/class/brand'].numpy().decode()
        color = features['image/class/color'].numpy().decode()
        model = features['image/class/model'].numpy().decode()
        year  = features['image/class/year'].numpy().decode()
        image = features['image/encoded']
        image = tf.io.decode_jpeg(image)
        image = tf.keras.preprocessing.image.array_to_img(image)

        return path, brand, color, color, model, year, image

    def show(self, parsed_tfrecord):
        self.parsed_tfrecord = parsed_tfrecord

        for features in self.parsed_tfrecord.take(self.n_image):          
            path, brand, color, color, model, year, image = \
            self._tensor_decode(features)          
            #plt.text(250, 0, 'path : '+ path)
            plt.text(250, 20, 'brand : '+ brand)
            plt.text(250, 40, 'color : '+ color)
            plt.text(250, 60, 'model : '+ model)
            plt.text(250, 80, 'year : '+ year)

            # plt.text(10.0, 1, 'brand : '+ brand)
            
            plt.imshow(image)
            plt.show()

    def resize_img(self, tensor):
        self.tensor = tensor
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord-path', type=str, dest='tfrecord_path',
                        default=_DEFAULT_TFRECORD_DIR,
                        help='relative path of the tfrecord path in the project file.'
                            f'(defaults: {_DEFAULT_TFRECORD_DIR})')
    parser.add_argument('--n-image', type=int, dest='n_image', default=1,
                    help='number of image to show TFRecord into.'
                         '(defaults: 1)')

    return parser.parse_args()

def main(args):
    loader = TFRecordLoader(args.tfrecord_path)
    viewer = TFRecordViewer(args.n_image)
                            
    parsed_tfrecord = loader.load()
    viewer.show(parsed_tfrecord)
    

if __name__ == '__main__':
    main(parse_args())
