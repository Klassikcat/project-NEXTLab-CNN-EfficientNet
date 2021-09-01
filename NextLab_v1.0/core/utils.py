from core.config import cfg
from tqdm import tqdm

import os
import glob
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import random
import PIL
from PIL import Image 

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_example(path, img_str, brand_name, color, model, year):
    """
    :param img_str: byte-encoded image, str
    :param brand_name: brand name of car, str
    :param color: car's color name, str
    :param model: car's model, str
    :param year: car's year, str
    :return: tf.train.Example file
    """
    feature = {'image/path': _bytes_feature(path),
               'image/encoded': _bytes_feature(img_str),
               'image/class/brand': _bytes_feature(brand_name),
               'image/class/color': _bytes_feature(color),
               'image/class/model': _bytes_feature(model),
               'image/class/year': _bytes_feature(year)
               }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def deserialize_example(serialized_string): 
    features = {
        'image/path': tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/brand': tf.io.FixedLenFeature((), tf.string),
        'image/class/color': tf.io.FixedLenFeature((), tf.string),
        'image/class/model': tf.io.FixedLenFeature((), tf.string),
        'image/class/year': tf.io.FixedLenFeature((), tf.string)
    }
    new_features = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
    }

    tmp = tf.io.parse_single_example(serialized_string, features) 
    example = tf.io.parse_single_example(serialized_string, new_features)

    image = example['image'] = decode_image(tmp['image/encoded'])
    label = example['label'] = tmp['image/class/brand']+'_'+tmp['image/class/model']+'_'+tmp['image/class/year']
    return image, label

def get_paths(path=str, target=str):
    if target == 'json' :
        __DIR = '/**/**/**.json'
    elif target == 'tfrecord' :
        __DIR = '/*.tfrecord'
    ##assert
    paths = path + __DIR
    paths = glob.glob(path + __DIR)
    return paths

def get_paths_tfrecord_shard(tfrecord_path, shard_id, shard_size):
    return os.path.join(tfrecord_path, f'{shard_id:03d}-{shard_size}.tfrecord')

def get_jsons(image_path=str, jsons_paths=list):
    json_dataframe = pd.DataFrame()
    for path in tqdm(jsons_paths):
        with open(path, "r", encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            json_data = json_data['car']
            data = json_data['attributes']
            data['image_path'] = image_path + '/' + json_data['imagePath']
            json_dataframe = json_dataframe.append(data, ignore_index=True)
    return json_dataframe

def get_img(image_path=str):
    files = tf.io.read_file(image_path)
    return tf.image.decode_image(files, channels=3)

def get_tfrecord(path=str):
    tfrecord = tf.data.TFRecordDataset(path, compression_type = 'GZIP')
    return tfrecord.map(parse_image_function)

def parse_image_function(example_proto):
    feature = {
        'image/path': tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/brand': tf.io.FixedLenFeature((), tf.string),
        'image/class/color': tf.io.FixedLenFeature((), tf.string),
        'image/class/model': tf.io.FixedLenFeature((), tf.string),
        'image/class/year': tf.io.FixedLenFeature((), tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature)

def tensor_decode(features):
    path  = features['image/path'].numpy().decode()
    brand = features['image/class/brand'].numpy().decode()
    color = features['image/class/color'].numpy().decode()
    model = features['image/class/model'].numpy().decode()
    year  = features['image/class/year'].numpy().decode()
    image = features['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = tf.keras.preprocessing.image.array_to_img(image)
    return path, brand, color, color, model, year, image

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, cfg.NEXTLAB.IMAGE_SIZE)
    return image

def write_tfrecord_file(df, indices, shard_path):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(shard_path, options=options) as out:
        for index in indices:
            file_path = df.image_path.iloc[index].encode()
            brand = df.brand.iloc[index].encode()
            color = df.color.iloc[index].encode()
            model = df.model.iloc[index].encode()
            year = df.year.iloc[index].encode()
            decoded_image = open(df.image_path.iloc[index], 'rb').read()          
            example = make_example(file_path, decoded_image, brand, color, model, year)                        
            out.write(example.SerializeToString())

def read_class_names(class_name_path = ' ', counter=False):
    names = {}
    with open(class_name_path, 'r', encoding='UTF8') as data:
        if counter == False :
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        else : 
            for name in data:
                names[name.strip('\n')] = 0
    return names

def write_class_names(tfrecord_path, class_name_path = ' '): 
    paths =  glob.glob(os.path.join(tfrecord_path, '*.tfrecord'))

    class_list = []
    for path in paths :
        dataset = tf.data.TFRecordDataset(path, compression_type = 'GZIP').map(deserialize_example).batch(128)
        for images, classes in tqdm(dataset):
            for class_name in classes :
                class_list.append(class_name.numpy().decode())

    class_list = set(class_list)
    class_list = list(class_list)
    class_list.sort()

    with open(class_name_path, 'w', encoding='UTF8') as data:
        for class_name in class_list :
            data.write(f'{class_name}\n')

def count_class_names(image_path, label_path, dataframe_path):
    try :
        dataframe = pd.read_csv(dataframe_path, encoding='euc-kr')
    except :
        paths = get_paths(label_path, 'json') 
        dataframe = get_jsons(image_path, paths)
        dataframe.to_csv(dataframe_path, mode='w',encoding='euc-kr', index=False)

    class_names = read_class_names(cfg.NEXTLAB.CLASSES, counter=True)

    print("""\
        =================================================\n\
        #               please wait...!!                #\n\
        #      program is running to count class...     #\n\
        =================================================\n
        """)

    for index in range(len(dataframe)) :
        brand = dataframe.brand.iloc[index]
        model = dataframe.model.iloc[index]
        year  = dataframe.year.iloc[index]
        class_name = brand + '_' + model + '_' + year
        
        if class_name in class_names :
            class_names[class_name] += 1
    class_dict = {}
    class_dict['NumOfClass'] = class_names
    with open(cfg.NEXTLAB.CLASSES_JSON, 'w', encoding='utf-8') as make_file:
        json.dump(class_dict, make_file, indent="\t", ensure_ascii=False)

def edit_json(path, index = -1):
    
    ##./data/dataset/image/label/**/**/**.jpg  => ##./data/dataset/train/label/**/**/**.json 
    load_path = path.replace('image','label').replace('.jpg',f'.json')

    ##원본 이미지일 경우 index는 기본값인 -1이므로 json의 경로는 바뀌지만 확장자 앞에 _aug+숫자 가 붙지 않습니다. 
    # ##./data/dataset_aug/train/label/**/**/**.json 
    if(index < 0) : 
        save_path = path.replace('dataset','dataset_aug').replace('image','label').replace('.jpg','.json')
    ##증강된 이미지일 경우 index는 외부에서 받아오며 json의 경로와 확장자 앞에 _aug+숫자 가 붙게 바뀝니다.
    ##./data/dataset_aug/train/label/**/**/**_aug{index}.json 
    else :           
        save_path = path.replace('dataset','dataset_aug').replace('image','label').replace('.jpg',f'_aug{index}.json') 

    ##경로와 확장자만 json으로 변경된 load_path를 이용해 증강한 이미지에 대응하는 json을 불러와 수정합니다.
    with open(load_path, "r", encoding='UTF8') as json_file:
        json_data = json.load(json_file)
        if (index < 0) :
            json_data['car']['imagePath'] = json_data['car']['imagePath'].replace(".jpg", '.jpg')
        else : 
            json_data['car']['imagePath'] = json_data['car']['imagePath'].replace(".jpg", f'_aug{index}.jpg')
        #해치백/쉐보레_대우/해치백_마티즈-16.jpg -> 해치백/쉐보레_대우/해치백_마티즈-16_aug.jpg
    
    ##수정한 json을 저장합니다.
    with open(save_path, "w", encoding='UTF8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)

def augmentaion(df, seq, n_aug):
    
    for index in range(len(df)):
        image_paths = df['image_path'].iloc[index]
        count = int(df['count'].iloc[index])

        image_paths_toAgu = image_paths
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 작은 경우 증강할 이미지 수 만큼 
        ## 원본 이미지 경로를 랜덤하게 뽑아 추가해줍니다.
        if (count < n_aug) :
            while (len(image_paths_toAgu) < n_aug) :
                image_paths_toAgu+=[random.choice(image_paths)]
        ## 클래스당 이미지 보유 수가 증강할 이미지의 수보다 클 경우 증강할 이미지 수 만큼 
        ## 원본 이미지 경로를 랜덤하게 뽑아 줄여줍니다.           
        elif (count > n_aug): 
            while (image_paths_toAgu == n_aug) : 
                image_paths_toAgu = image_paths_toAgu.pop(random.randint(len(image_paths_toAgu)))
                image_paths_toAgu = set(image_paths_toAgu)
                image_paths_toAgu = list(image_paths_toAgu)
            count=n_aug

        ##원본 이미지를 dataset에서 열어서 dataset_aug폴더에 저장합니다.
        for path in image_paths_toAgu[:count] :
            image = np.array(PIL.Image.open(path))
            image = Image.fromarray(image)
            image.save(path.replace('dataset','dataset_aug'))
            edit_json(path)

        ##가지고 있는 이미지 수가 100장이 넘지 않는 경우에만 이미지를 증강합니다.
        if count !=  n_aug : 
            image_list=[]
            for path in tqdm(image_paths_toAgu[count:]) :
                image =  np.array(PIL.Image.open(path))
                image_list.append(image)

            #이미지를 증강합니다.
            images_aug = seq(images=image_list)

            #증강한 이미지를 저장합니다 경로명은 dataset에서 dataset_aug으로 변경하고 확장자 앞에 _aug+숫자 를 붙여줍니다.
            for i, image in enumerate(images_aug) :
                image = Image.fromarray(image)
                path = image_paths_toAgu[i]  ##./data/dataset/train/image/**/**/**.jpg
                image.save(path.replace('dataset','dataset_aug').replace('.jpg',f'_aug{i}.jpg')) ##./data/dataset_aug/train/image/**/**/**_aug_{i}.jpg    
                #증강한 이미지에 대응하는 json파일을 수정 및 저장합니다.
                edit_json(path, i)
        
        print(f"{index} :: {df['class_name'].iloc[index]} :: ImagePath is done!")
