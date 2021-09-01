from core import utils
from core.config import cfg

from imgaug import augmenters as iaa
from argparse import RawTextHelpFormatter

import argparse
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Please enter augument image number',
                                     formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--n_augumentaion', required=True, type=int, dest='n_aug',
                        help= 'number of augument')
    return parser.parse_args()

def main(args):
    ##Agumentaion 시퀀셜을 설정합니다.
    seq = iaa.Sequential([
                      iaa.Cutout(nb_iterations=2),
                      iaa.Affine(rotate=(-25, 25)),
                      iaa.Fliplr(0.5),
                      iaa.GammaContrast((0.5, 2.0)),
                        ])

    ##"./data/classes/car.json"의 경로에서 car.json (클래스당 카운팅 된 json 파일)을 불러옵니다.
    
    with open(cfg.NEXTLAB.CLASSES_JSON, "r", encoding='UTF8') as json_file:
        json_data = json.load(json_file)
        class_names = json_data['NumOfClass']

    df_classes = pd.DataFrame()
    df_classes['class_name'] = class_names.keys()
    df_classes['count'] = class_names.values()
    df_classes['image_path'] = ''

    ## "./data/classes/json_paths.csv"의 경로에서 이미지 정보가 저장된 csv를 로드합니다
    ## car.json와 json_paths.csv 병합해 클래스, 클래스당 개수, 클래스에 해당하는 원본 이미지의 경로를 가지게 하는 DataFrame을 생성합니다.
    df_json_paths = pd.read_csv(cfg.NEXTLAB.JSON_PATHS_DATAFRAME, encoding='euc-kr')
    for index in range(len(df_classes)):
        filterd_df = df_json_paths[df_json_paths['class_name']==df_classes['class_name'].iloc[index]]
        filterd_df = filterd_df['image_path']
        df_classes['image_path'].iloc[index] = filterd_df.tolist()
    
    ##Aunmentaion을 진행합니다.
    utils.augmentaion(df_classes, seq, args.n_aug)

if __name__ == '__main__':

    main(parse_args())