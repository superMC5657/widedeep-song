import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_song_dataset(data_path, read_part=False, sample_num=10000, test_size=0.2, embed_dim=4):
    """
    a example about creating song dataset'
    :param data_path: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    if read_part:
        data_df = pd.read_csv(data_path, iterator=True)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(data_path)

    # 离散特征: 专辑链接（字符串）、艺术家链接（字符串）、曲调(0-12)、音符时值（0-5）、所属歌单（整数连续） /曲目链接（字符串）"track_uri" 不使用 "pid"  不使用"time_signature"
    embedding_feature_items = ['key']
    raw_feature_items = ['mode']
    sparse_feature_items = embedding_feature_items + raw_feature_items

    # 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
    dense_feature_items = ["duration_ms_x", "acousticness", "danceability", 'energy', 'instrumentalness', 'liveness',
                           'loudness', 'speechiness', 'tempo', 'valence']  # continuous

    # 标签 multi-hot
    label_items = 'pid'

    # feature columns
    dense_feature_columns = [denseFeature(item) for item in dense_feature_items]
    embedding_feature_columns = [sparseFeature(item, len(data_df[item].unique()), embed_dim=embed_dim) for item in
                                 embedding_feature_items]
    raw_feature_columns = [denseFeature(item) for item in raw_feature_items]

    # 稠密特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_feature_items] = mms.fit_transform(data_df[dense_feature_items])

    # 对稀疏特征做labelEncoder
    le = LabelEncoder()
    for item in sparse_feature_items:
        data_df[item] = le.fit_transform(data_df[item])

    def construct_data(data):
        dense_features = data[dense_feature_items].values.astype('float32')
        embedding_features = data[embedding_feature_items].values.astype('int32')
        raw_features = data[raw_feature_items].values.astype('int32')
        label = data[label_items]
        return (dense_features, embedding_features, raw_features), indice_2_multi(label)

    train_data, test_data = train_test_split(data_df, test_size=test_size)
    train_x, train_y = construct_data(train_data)
    test_x, test_y = construct_data(test_data)
    return (dense_feature_columns, embedding_feature_columns, raw_feature_columns), (train_x, train_y), (test_x, test_y)


def indice_2_multi(label):
    size = label.shape[0]
    multi_hot_label = np.zeros((size, 3000))
    for index, item in enumerate(label):
        m_hot = np.zeros(shape=3000, dtype=np.int)
        l = np.array([int(_) for _ in item.split(',')])
        m_hot[l] = 1
        multi_hot_label[index, :] = m_hot
    return multi_hot_label


if __name__ == "__main__":
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_path = 'data/groupby.csv'
    create_song_dataset(data_path, read_part=True)
    # label = "[1 2 3 4]"
    # indice_2_multi(label)
