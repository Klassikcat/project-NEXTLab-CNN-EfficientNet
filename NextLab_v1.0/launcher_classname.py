from core import utils
from core.config import cfg
from argparse import RawTextHelpFormatter

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Please select a function and enter it',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--function', required=True, type=str, dest='func',
                        help= '''
                        'count' : count class name
                        'write' : write class name
                        ''')
    return parser.parse_args()

def main(args):
    FUNC_FLAG = args.func.upper()

    if FUNC_FLAG == 'COUNT' :
        utils.count_class_names(cfg.NEXTLAB.TRAIN_IMAGES_PATH,
                                cfg.NEXTLAB.TRAIN_LABELS_PATH,
                                cfg.NEXTLAB.JSON_PATHS_DATAFRAME)

    elif FUNC_FLAG == 'WRITE' :     
        utils.write_class_names(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH,
                                cfg.NEXTLAB.CLASSES)
    else :
        print("Please selcect again!!")

    print("Done!!")

if __name__ == '__main__':

    main(parse_args())