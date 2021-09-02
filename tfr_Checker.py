from core.config import cfg
from tqdm import tqdm

from argparse import RawTextHelpFormatter

import argparse
import os
import glob
import tensorflow as tf

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
    return parser.parse_args()

def get_length(file_name):
        tmp = file_name.split('-')
        tmp = tmp[1].split('.')
        return int(tmp[0])

def check_tfrecord(path):
    total_images = 0
    tfrecord_pathes = sorted(glob.glob(os.path.join(path, '*')))
    print("=========================================================================================================================")
    print("#                                                      Checking...                                                      #")
    print("=========================================================================================================================") 
       
    for idx, tfrecord_path in enumerate(tfrecord_pathes):
        fname, ext = os.path.splitext(tfrecord_path)
        assert ext == '.tfrecord', 'it is not a tfrecord file.'
        try:
            total_images += sum([1 for _ in tqdm(tf.data.TFRecordDataset(tfrecord_path, compression_type = 'GZIP'),
                                                                         total = get_length(tfrecord_path))]) # Check corrupted tf records
            print("{}: {} is ok".format(idx, tfrecord_path))
        except:
            print("{}: {} is corrupted".format(idx, tfrecord_path))
    print("Succeed, no corrupted tf records found for {} images".format(total_images))


def main(args):
    FUNC_FLAG = args.func.upper()

    if FUNC_FLAG == 'TRAIN' :
        check_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH)

    elif FUNC_FLAG == 'VALID' :     
        check_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH)

    elif FUNC_FLAG == 'ALL' :
        check_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH)

        check_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH)
    else :
        print("Please selcect again!!")

    print("Done!!")

if __name__ == '__main__':

    main(parse_args())