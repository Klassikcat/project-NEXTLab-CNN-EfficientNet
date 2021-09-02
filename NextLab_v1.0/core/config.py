#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# NEXTLAB options
__C.NEXTLAB                      = edict()

__C.NEXTLAB.CLASSES              = "./data/classes/car.names"
__C.NEXTLAB.CLASSES_JSON         = "./data/classes/car.json"
__C.NEXTLAB.JSON_PATHS_DATAFRAME = "./data/classes/json_paths.csv"
__C.NEXTLAB.TRAIN_IMAGES_PATH    = "./data/dataset/train/image"
__C.NEXTLAB.TRAIN_LABELS_PATH    = "./data/dataset/train/label"
__C.NEXTLAB.TRAIN_TFRECORDS_PATH = "./data/dataset/train/tfrecord"
__C.NEXTLAB.VALID_IMAGES_PATH    = "./data/dataset/valid/image"
__C.NEXTLAB.VALID_LABELS_PATH    = "./data/dataset/valid/label"
__C.NEXTLAB.VALID_TFRECORDS_PATH = "./data/dataset/valid/tfrecord"

__C.NEXTLAB.N_SHARDS             = 1
__C.NEXTLAB.SEED                 = 42
__C.NEXTLAB.IMAGE_SIZE           = [224,224]
__C.NEXTLAB.BATCH_SIZE           = 128