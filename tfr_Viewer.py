from core import utils
from core.config import cfg
from argparse import RawTextHelpFormatter

import argparse
import matplotlib.pyplot as plt
import platform

def parse_args():
    parser = argparse.ArgumentParser(description='Please select a function and enter it',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--function', required=True, type=str, dest='func',
                        help= '''
                        'train' : show tfrecord only for TRAIN dataset
                        'valid' : show tfrecord only for VALIDATION dataset
                        'all'   : show tfrecord only for TRAIN & VALIDATION dataset
                        ''')
    parser.add_argument('--nimage', default = 1, type=int, dest='n_image', 
                        help= 'number of showing image per each tfrecord. (default = 1)')
    return parser.parse_args()

def show_tfrecord(tfrecord_path, n_image):
    paths = utils.get_paths(tfrecord_path, 'tfrecord')
    count =0
    for path in paths :
        parsed_tfrecord = utils.get_tfrecord(path)
        #for features in parsed_tfrecord.take(n_image) :
        for features in parsed_tfrecord :
            path, brand, model, year, image, class_name, label = \
            utils.tensor_decode(features)
            if count == 32100:
                print(path)
                print(brand)
                print(model)
                print(year)
                print(class_name)
                print(str(label.numpy()))
                plt.text(180, 0, 'path : '+ path)
                plt.text(180, 10, 'brand : '+ brand)
                plt.text(180, 20, 'model : '+ model)
                plt.text(180, 30, 'year : '+ year)
                plt.text(180, 40, 'class_name : '+ class_name) 
                plt.text(180, 50, 'label : '+ str(label.numpy()))
                plt.imshow(image)
                plt.show() 
            count+=1
            print(count)
    print(count)
def main(args):
    FUNC_FLAG = args.func.upper()

    if FUNC_FLAG == 'TRAIN' :
        show_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                      args.n_image)

    elif FUNC_FLAG == 'VALID' :     
        show_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                      args.n_image)

    elif FUNC_FLAG == 'ALL' :
        show_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                      args.n_image)

        show_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                      args.n_image)
    else :
        print("Please selcect again!!")

    print("Done!!")

if __name__ == '__main__':

    if platform.system() == 'Darwin': #Mac
            plt.rc('font', family='AppleGothic') 
    elif platform.system() == 'Windows': #Window
        plt.rc('font', family='Malgun Gothic') 
    elif platform.system() == 'Linux': #Linux or Colab
        plt.rc('font', family='Malgun Gothic') 
    plt.rcParams['axes.unicode_minus'] = False #resolve minus symbol breaks when using Hangul font

    main(parse_args())
