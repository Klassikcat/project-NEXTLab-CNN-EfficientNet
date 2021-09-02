from core import utils
from core.config import cfg
from argparse import RawTextHelpFormatter

import argparse
import numpy as np
import math
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Please select a function and enter it',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--function', required=True, type=str, dest='func',
                        help= '''
                        'train' : generate tfrecord only for TRAIN dataset
                        'valid' : generate tfrecord only for VALIDATION dataset
                        'all'   : generate tfrecord only for TRAIN & VALIDATION dataset
                        ''')
    return parser.parse_args()

def generate_tfrecord(image_path, label_path, tfrecord_path, dataframe_path, n_shards):
    try :
        dataframe = pd.read_csv(dataframe_path, encoding='euc-kr')
    except :
        paths = utils.get_paths(label_path, 'json') 
        dataframe = utils.get_jsons(image_path, paths)
        dataframe.to_csv(dataframe_path, mode='w',encoding='euc-kr', index=False)

    size = len(dataframe)
    offset = 0
    shard_size = math.ceil(size / n_shards)
    cumulative_size = offset + size

    print("""\
        =================================================\n\
        #               please wait...!!                #\n\
        #   program is running to generate tfrecord...  #\n\
        =================================================\n
        """)

    for shard_id in range(1, n_shards + 1):
        step_size = min(shard_size, cumulative_size - offset)
        shard_path = utils.get_paths_tfrecord_shard(tfrecord_path, shard_id, step_size)
        file_indices = np.arange(offset, offset + step_size)
        utils.write_tfrecord_file(dataframe, file_indices, shard_path)
        offset += step_size
 
def main(args):
    FUNC_FLAG = args.func.upper()

    if FUNC_FLAG == 'TRAIN' :
        generate_tfrecord(cfg.NEXTLAB.TRAIN_IMAGES_PATH,
                          cfg.NEXTLAB.TRAIN_LABELS_PATH,
                          cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                          cfg.NEXTLAB.JSON_PATHS_DATAFRAME,
                          cfg.NEXTLAB.N_SHARDS)

    elif FUNC_FLAG == 'VALID' :     
        generate_tfrecord(cfg.NEXTLAB.VALID_IMAGES_PATH,
                          cfg.NEXTLAB.VALID_LABELS_PATH,
                          cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                          cfg.NEXTLAB.JSON_PATHS_DATAFRAME,
                          cfg.NEXTLAB.N_SHARDS)
 
    elif FUNC_FLAG == 'ALL' :
        generate_tfrecord(cfg.NEXTLAB.TRAIN_IMAGES_PATH,
                          cfg.NEXTLAB.TRAIN_LABELS_PATH,
                          cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                          cfg.NEXTLAB.JSON_PATHS_DATAFRAME,
                          cfg.NEXTLAB.N_SHARDS)

        generate_tfrecord(cfg.NEXTLAB.VALID_IMAGES_PATH,
                          cfg.NEXTLAB.VALID_LABELS_PATH,
                          cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                          cfg.NEXTLAB.JSON_PATHS_DATAFRAME,
                          cfg.NEXTLAB.N_SHARDS)
    else :
        print("Please selcect again!!")

    print("Done!!")

if __name__ == '__main__':
    main(parse_args())
