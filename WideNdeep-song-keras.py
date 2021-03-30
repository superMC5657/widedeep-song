import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from tensorflow.keras.experimental import WideDeepModel
from tensorflow.keras.losses import binary_crossentropy, MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

def create_song_dataset(file, embed_dim=8, read_part=True, sample_num=10000, test_size=0.2):
    """
    a example about creating song dataset'
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    # 读取
    selected_features = ['album_uri', 'artist_uri', 'pid', 'acousticness', 'danceability', 'energy', 'instrumentalness',
                         'key',
                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    if read_part:
        data_df = pd.read_csv(file, iterator=True)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(file)
    # selected_df = data_df[selected_features]
    # 区分稠密稀疏特征
    sparse_features = ["album_uri", "artist_uri", "pid", "key", "track_uri"]  # categorical，track_uri 加入 pid是label 分开处理
    dense_features = ["acousticness", "danceability", 'energy', 'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo', 'valence']  # continuous
    # label = ["pid"]
    # sparse_data = data_df[sparse_features]
    # dense_data = data_df[dense_features]

    # 对稀疏特征做labelEncoder
    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 特征工程
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])
    # 尝试把feature放到一个尺度
    data_df[sparse_features] = mms.fit_transform(data_df[sparse_features])
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]
    train, test = train_test_split(data_df, test_size=test_size)

    train_x = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
    train_y = train['pid'].values.astype('int32')

    test_x = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
    test_y = test['pid'].values.astype('int32')
    #view_test_x = pd.DataFrame(test_x)
    return feature_columns, (train_x, train_y), (test_x, test_y)