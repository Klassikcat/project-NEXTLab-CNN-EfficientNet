from core import utils
from core.config import cfg
from argparse import RawTextHelpFormatter

import argparse
import numpy as np
import math
import pandas as pd
##validation관련해서 제대로 생성되는지에 대한 이슈 해결

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

def generate_tfrecord(tfrecord_path, n_shards, FUCTION_FLAG=True):
    ##classes 폴더에 저장된 car.json폴더를 읽어옵니다.
    dataframe = pd.read_json(cfg.NEXTLAB.CLASSES_JSON, "r", encoding='UTF8')
    dataframe = dataframe.T.rename_axis('class_name').reset_index()

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
        utils.write_tfrecord_file(dataframe, file_indices, shard_path, FUCTION_FLAG)
        offset += step_size
 
def main(args):
    FUNC_FLAG = args.func.upper()
    if FUNC_FLAG == 'TRAIN' :
        generate_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                          cfg.NEXTLAB.N_SHARDS,
                          True)

    elif FUNC_FLAG == 'VALID' :     
        generate_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                          cfg.NEXTLAB.N_SHARDS,
                          False)
 
    elif FUNC_FLAG == 'ALL' :
        generate_tfrecord(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                          cfg.NEXTLAB.N_SHARDS,
                          True)

        generate_tfrecord(cfg.NEXTLAB.VALID_TFRECORDS_PATH,
                          cfg.NEXTLAB.N_SHARDS,
                          False)
    else :
        print("Please selcect again!!")

    print("Done!!")
#python tfr_Generator.py --function all
if __name__ == '__main__':
    main(parse_args())
