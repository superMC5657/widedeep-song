import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def dataset_feature_engineer(data_df, sparse_feature_items, dense_feature_items, embed_dim=8):
    # 复制一个专门的df做特征工程
    data_feature_engineer = data_df.copy()
    # 稠密特征归一化 有些特征已经是0-1之间 有些不是 对那些不是对做归一化
    #normalizer_dense_feature = ["duration_ms_x", "loudness", "tempo"]
    #mms = MinMaxScaler(feature_range=(0, 1))
    #data_feature_engineer[dense_feature_items] = mms.fit_transform(data_feature_engineer[dense_feature_items]) # 测试后，（0-1）基本没有发生变化 不在（0-1）的值归一化了
    duration_ms_x = tf.feature_column.numeric_column("duration_ms_x")
    acousticness = tf.feature_column.numeric_column("acousticness")
    danceability = tf.feature_column.numeric_column("danceability")
    energy = tf.feature_column.numeric_column("energy")
    instrumentalness = tf.feature_column.numeric_column("instrumentalness")
    liveness = tf.feature_column.numeric_column("liveness")
    loudness = tf.feature_column.numeric_column("loudness")
    speechiness = tf.feature_column.numeric_column("speechiness")
    tempo= tf.feature_column.numeric_column("tempo")
    valence = tf.feature_column.numeric_column("valence")
    dense_feature_column = [duration_ms_x, acousticness, danceability, energy, instrumentalness, liveness, loudness,
                      speechiness, tempo, valence]
    dense_features = tf.compat.v1.feature_column.input_layer(data_feature_engineer[dense_feature_items].to_dict(), dense_feature_column)
    dense_features
    # 稀疏特征 由于
    # 获取 稀疏特征的 one-hot 稀疏特征可能
    #unique_sparse_feature = {feature: len(data_feature_engineer[feature].unique()) for feature in sparse_feature_items}
    #album_uri =
    #artist_uri =
    #key =
    #time_signature =
    #mode =
    #pid =



def create_song_dataset(data, read_part=True, sample_num=10000, test_size=0.2):
    """
    a example about creating song dataset'
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    if read_part:
        data_df = pd.read_csv(data, iterator=True)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(data)
    # 离散特征: 专辑链接（字符串）、艺术家链接（字符串）、曲调(0-12)、音符时值（0-5）、所属歌单（整数连续） /曲目链接（字符串）"track_uri" 不使用
    sparse_feature_items = ["album_uri", "artist_uri", "key", "time_signature", "mode", "pid"]
    # 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
    dense_feature_items = ["duration_ms_x", "acousticness", "danceability", 'energy', 'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo', 'valence']  # continuous
    sparse_features, dense_features = dataset_feature_engineer(data_df, sparse_feature_items, dense_feature_items)
    # tf.dataloader

if __name__ == "__main__":
    data_path = '/Myhome/zy/workspace/rs/widedeep-song/data/data_sample.csv'
    create_song_dataset(data_path)