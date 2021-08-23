import tensorflow as tf
import glob
import argparse
import os
from tqdm.auto import tqdm

_DEFAULT_TFRECORD_DIR = 'tfrecord' 
def get_length(file_name):
    tmp = file_name.split('-')
    tmp = tmp[1].split('.')
    return int(tmp[0])

def check_tfrecord(origin_path, tfrecord_path):
    total_images = 0
    tfrecord_path = sorted(glob.glob(os.path.join(origin_path, 
                                                  tfrecord_path, 
                                                  '*')))
    for idx, path in enumerate(tfrecord_path):
        fname, ext = os.path.splitext(path)
        assert ext == '.tfrecord', 'it is not a tfrecord file.'
        try:
            total_images += sum([1 for _ in tqdm(tf.data.TFRecordDataset(path, compression_type = 'GZIP'),\
                                                                         total = get_length(path))]) # Check corrupted tf records
            print("{}: {} is ok".format(idx, path))
        except:
            print("{}: {} is corrupted".format(idx, path))

    print("Succeed, no corrupted tf records found for {} images".format(total_images))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-path', type=str, dest='origin_path',
                        help='Absolute path of the target path.'
                             '(example = /User/administrator/project-path/contents/)')
    parser.add_argument('--tfrecord-path', type=str, dest='tfrecord_path',
                        default=_DEFAULT_TFRECORD_DIR,
                        help='relative path of the tfrecord path in the project file.'
                            f'(defaults: {_DEFAULT_TFRECORD_DIR})')

    return parser.parse_args()

def main(args):
    check_tfrecord(args.origin_path, args.tfrecord_path)

if __name__ == '__main__':
    main(parse_args())