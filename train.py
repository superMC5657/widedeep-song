import argparse
import warnings

from tensorflow import keras

from models.widedeep import WideDeep
from utils.dataset import create_song_dataset
from utils.base import generate_dir, get_cfg

warnings.filterwarnings("ignore")
import os
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall


def train():
    # 数据处理
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    train_cfg, model_cfg, dataset_cfg, inf_cfg, work_dir, model_dir = get_cfg(args)
    generate_dir(work_dir)
    feature_columns, train, test = create_song_dataset(data_path=dataset_cfg.get('data_path'),
                                                       dataset_cfg=dataset_cfg,
                                                       read_part=train_cfg.get('read_part'),
                                                       sample_num=train_cfg.get('sample_num'),
                                                       test_size=train_cfg.get('test_size'),
                                                       embed_dim=model_cfg.get('embed_dim'))
    train_X, train_y = train
    test_X, test_y = test
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model, embed_model = WideDeep(feature_columns,
                                  model_cfg.get('deep_hidden_units'),
                                  model_cfg.get('wide_hidden_units'),
                                  label_embed_nums=model_cfg.get('label_embed_nums'),
                                  dnn_dropout=model_cfg.get('dnn_dropout'),
                                  classes=dataset_cfg.get('classes')
                                  ).create_model()
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=train_cfg.get('lr')),
                  metrics=[AUC(), Recall()])

    # ===========================Fit==============================

    model.fit(
        train_X,
        train_y,
        epochs=train_cfg.get('epochs'),
        # callbacks=[EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True)],
        # checkpoint
        batch_size=train_cfg.get('batch_size'),
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=train_cfg.get('batch_size'))[1])
    # ===========================Save==============================
    model.save(filepath=os.path.join(model_dir, 'model'))
    embed_model.save(filepath=os.path.join(model_dir, 'embed_model'))
    keras.utils.plot_model(model, to_file=os.path.join(work_dir, 'model.png'))
    keras.utils.plot_model(embed_model, to_file=os.path.join(work_dir, 'embed_model.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()
