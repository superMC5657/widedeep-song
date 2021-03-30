import warnings
warnings.filterwarnings("ignore")
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy, MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from tensorflow.keras.experimental import WideDeepModel # 使用keras的模型


# from utils import SparseFeat, DenseFeat, VarLenSparseFeat
class Linear(Layer):
    """
    Linear Part
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result

class DNN(Layer):
    """
	Deep Neural Network
	"""

    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        """
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)

        # Wide
        wide_out = self.linear(dense_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs])).summary()

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


if __name__ == "__main__":
    # 数据处理
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_path = '/Myhome/zy/workspace/rs/widedeep-song/data/data_sample.csv'
    read_part = True
    sample_num = 50000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    feature_columns, train, test = create_song_dataset(file=data_path,
                                                       embed_dim=embed_dim,
                                                       read_part=read_part,
                                                       sample_num=sample_num,
                                                       test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = WideDeep(feature_columns, hidden_units)
        #model.summary()
        # =========================Compile============================
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), #
                      metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
