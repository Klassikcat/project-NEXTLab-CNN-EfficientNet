from core.config import cfg
from core import utils
import tensorflow as tf
import os
import glob
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np



IMAGE_SIZE = [224,224]
BATCH_SIZE = 32
EPOCHS = 5
AUTO = tf.data.experimental.AUTOTUNE

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label  

def get_training_dataset():
    path =  glob.glob(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH + '/*.tfrecord')
    dataset = tf.data.TFRecordDataset(path, compression_type = 'GZIP')
    dataset = dataset.map(utils.deserialize_example)
    dataset = dataset.map(data_augment)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    path =  glob.glob(cfg.NEXTLAB.VALID_TFRECORDS_PATH + '/*.tfrecord')
    dataset = tf.data.TFRecordDataset(path, compression_type = 'GZIP')
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def main(): 
    train_path =  glob.glob(cfg.NEXTLAB.TRAIN_TFRECORDS_PATH + '/*.tfrecord')
    valid_path =  glob.glob(cfg.NEXTLAB.VALID_TFRECORDS_PATH + '/*.tfrecord')

    NUM_TRAINING_IMAGES = utils.get_length(train_path[0])
    NUM_VALIDATION_IMAGES = utils.get_length(valid_path[0])
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

    print('Dataset: {} training images, {} validation images'
          .format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))


    enb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    enb0.trainable = True # Full Training
    
    model = tf.keras.Sequential([
        enb0,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(322, activation='softmax')
    ])
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    
    model.summary()
    
    historyefnet = model.fit(get_training_dataset(), 
                             steps_per_epoch=STEPS_PER_EPOCH, 
                             epochs=EPOCHS, 
                             validation_data=get_validation_dataset())


if __name__ == "__main__": 
    main()


