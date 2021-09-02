import argparse
import tfrecordUtils as tfutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FUNCTION', type=str, dest='func',
                        default=_DEFAULT_FUNC_FLAG,
                        help='Enter the function want to execute.'\
                            'functions : view, check, aug'),
    parser.add_argument('--tfrecord-path', type=str, dest='tfrecord_path',
                        default=_DEFAULT_TFRECORD_DIR,
                        help='Absolute path of the saved data path.'
                             '(example = /User/administrator/project-path/contents/)'
                             f'(defaults: {_DEFAULT_TFRECORD_DIR})'),
    parser.add_argument('--file-name', type=str, dest='file_name',
                        default=_DEFAULT_TARGET_FILE,
                        help='Name of the target tfrecord.'
                            f'(defaults: {_DEFAULT_TARGET_FILE})')
    parser.add_argument('--n-image', type=int, dest='n_image', 
                        default=1,
                        help='number of image to show TFRecord into.'
                         '(defaults: 1)')

    return parser.parse_args()

def view(args):
    loader = tfutils.TFRecordLoader(args.tfrecord_path, args.file_name)
    viewer = tfutils.TFRecordViewer(args.n_image)
                            
    parsed_tfrecord = loader.load()
    viewer.show(parsed_tfrecord)

def check(args):
    checker = tfutils.TFRecordChecker(args.tfrecord_path)
    checker.check_tfrecord()

_DEFAULT_FUNC_FLAG = 'VIEW'                                                ##경로 수정 필요
_DEFAULT_TFRECORD_DIR = 'C:/Users/Ung/Desktop/NextLab/data/train/tfrecord' ##경로 수정 필요
_DEFAULT_TARGET_FILE = '001-735833.tfrecord'                               ##경로 수정 필요

if __name__ == '__main__':
    FUNC_FLAG = parse_args().func.upper()
    if FUNC_FLAG == 'VIEW' :
        view(parse_args())
    elif FUNC_FLAG == 'CHECK' :      
        check(parse_args())