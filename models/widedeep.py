# -*- coding: utf-8 -*-
# !@time: 2021/3/29 下午9:47
# !@author: superMC @email: 18758266469@163.com
# !@fileName: widedeep.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding


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
    def __init__(self, feature_columns, deep_hidden_units, wide_hidden_units, activation='relu',
                 label_embed_nums=512, classes=3000,
                 dnn_dropout=0.1, embed_reg=1e-4):
        """
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param label_embed_nums: A int.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """

        super(WideDeep, self).__init__()
        self.embed_model = EmbedModel(feature_columns, deep_hidden_units, wide_hidden_units, activation=activation,
                                      label_embed_nums=label_embed_nums, dnn_dropout=dnn_dropout, embed_reg=embed_reg)

        self.dense_feature_columns, self.embedding_feature_columns, self.raw_feature_columns = feature_columns
        # classes_num binary classification
        self.final_linear = Dense(classes)

    def call(self, inputs, **kwargs):
        dense_inputs, embedding_inputs, raw_inputs = inputs
        out_embedding = self.embed_model(inputs)
        output = self.final_linear(out_embedding)
        # out
        output = tf.nn.sigmoid(output)
        return output

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        embedding_inputs = Input(shape=(len(self.embedding_feature_columns),), dtype=tf.int32)
        raw_inputs = Input(shape=(len(self.raw_feature_columns, )), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, embedding_inputs, raw_inputs],
                       outputs=self.call([dense_inputs, embedding_inputs, raw_inputs])).summary()


class EmbedModel(tf.keras.Model):
    def __init__(self, feature_columns, deep_hidden_units, wide_hidden_units, activation='relu', label_embed_nums=512,
                 dnn_dropout=0.1, embed_reg=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_feature_columns, self.embedding_feature_columns, self.raw_feature_columns = feature_columns
        # sparse feature embedding
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.embedding_feature_columns)
        }
        # dense network
        self.deep_network = DNN(deep_hidden_units, activation, dnn_dropout)
        # sparse network
        self.wide_network = DNN(wide_hidden_units, activation, dnn_dropout)

        # label_embedding
        self.label_embedding = Dense(label_embed_nums)

    def call(self, inputs, **kwargs):
        dense_inputs, embedding_inputs, raw_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](embedding_inputs[:, i])
                                  for i in range(embedding_inputs.shape[1])], axis=-1)
        raw_inputs = tf.squeeze(tf.one_hot(raw_inputs, depth=2), axis=1)
        x = tf.concat([dense_inputs, sparse_embed, raw_inputs], axis=-1)

        # Wide
        wide_out = self.wide_network(dense_inputs)
        # Deep
        deep_out = self.deep_network(x)
        out = tf.concat([wide_out, deep_out], axis=-1)
        out_embedding = self.label_embedding(out)

        return out_embedding

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        embedding_inputs = Input(shape=(len(self.embedding_feature_columns),), dtype=tf.int32)
        raw_inputs = Input(shape=(len(self.raw_feature_columns, )), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, embedding_inputs, raw_inputs],
                       outputs=self.call([dense_inputs, embedding_inputs, raw_inputs])).summary()


