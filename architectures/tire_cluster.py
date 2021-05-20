import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
if os.environ["CUDA_VISIBLE_DEVICES"] == "0":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Layer, Multiply
from tensorflow.keras.models import Model
import numpy as np


class Cust_loss(Layer):
    def __init__(self, **kwargs):
        super(Cust_loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # split inputs
        x, y_true, y_pred, x_decoded, z_shared = inputs

        # define losses
        # reconstruction loss
        squared_diff = K.square(x - x_decoded)
        mse_loss = tf.reduce_mean(squared_diff)
        # shared loss
        aeqb = (tf.equal(y_true[:, 0, :], y_true[:, 1, :]))
        aeqb_int = tf.cast(aeqb, tf.int32)
        result1 = tf.equal(tf.reduce_sum(aeqb_int, axis=1), tf.reduce_sum(tf.ones_like(aeqb_int), axis=1))
        z_same = tf.squeeze(tf.gather(z_shared, tf.where(result1)), [1])
        square_diff2 = tf.reduce_mean((z_same[:, 1:, :] - z_same[:, :1, :]) ** 2)
        # TODO: add loss for different classes
        # result2 = tf.not_equal(tf.reduce_sum(aeqb_int, axis=1), tf.reduce_sum(tf.ones_like(aeqb_int), axis=1))
        # z_diff = tf.squeeze(tf.gather(z_shared, tf.where(result2)), [1])
        # square_diff3 = tf.reduce_mean((z_diff[:, 1:, :] - z_diff[:, :1, :]) ** 2)
        shared_loss = kwargs['loss_weight_share'] * tf.math.maximum(square_diff2,0)
        if tf.math.is_nan(shared_loss):
            shared_loss = 0.0
        # classification loss
        cce = keras.losses.CategoricalCrossentropy(label_smoothing=0)
        ce_loss = kwargs['loss_weight_CE'] * cce(y_true, y_pred)

        # add losses to model
        self.add_loss(mse_loss, inputs=True)
        self.add_metric(mse_loss, aggregation="mean", name="mse_loss")
        self.add_loss(ce_loss, inputs=True)
        self.add_metric(ce_loss, aggregation="mean", name="ce_loss")
        self.add_loss(shared_loss, inputs=True)
        self.add_metric(shared_loss, aggregation="mean", name="shared_loss")
        total_loss = mse_loss + shared_loss + ce_loss
        return total_loss


def create_parallel_aes(window_size_per_ae,
                        label_size_per_ae,
                        labels,
                        intermediate_dim=0,
                        latent_dim=1,
                        nr_ae=3,
                        nr_spatial=1,
                        nr_temporal=1,
                        loss_weight_share=1,
                        loss_weight_CE=1):
    """
    Create a Tensorflow model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.

    Args:
        window_size_per_ae: window size for the AE
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_ae: number of parallel AEs (K in paper)
        nr_shared: number of shared features (should be <= latent_dim)
        loss_weight: lambda in paper

    Returns:
        A parallel AE model instance, its encoder part and its decoder part
    """
    nr_unshared = latent_dim - (nr_spatial + nr_temporal)
    wspa = window_size_per_ae
    lspa = label_size_per_ae
    x = Input(shape=(nr_ae, wspa,), name='data')
    lbl = Input(shape=(nr_ae, lspa,), name='lbl')
    if intermediate_dim == 0:
        y = x
    else:
        y = Dense(intermediate_dim, activation=tf.nn.tanh, name='dense_1')(x)
        # y = tf.keras.layers.BatchNormalization()(y)

    z_spatial = Dense(nr_spatial, activation=tf.nn.tanh, name='dense_2-1')(y)
    z_temporal = Dense(nr_temporal, activation=tf.nn.tanh, name='dense_2-2')(y)
    z_unshared = Dense(nr_unshared, activation=tf.nn.tanh, name='dense_2-3')(y)
    z_shared = tf.concat([z_spatial, z_temporal], -1)
    z = tf.concat([z_shared, z_unshared], -1)
    classes = labels.max() + 1
    l = Dense(classes, activation='softmax', name='dense_5')(z_shared)

    if (nr_spatial != 1) and (nr_spatial != 0) and (nr_unshared != 0):
        splits = tf.split(z_unshared, num_or_size_splits=nr_unshared, axis=2)
        for idx, split in enumerate(splits):
            multi = tf.tile(split, multiples=[1, 1, nr_spatial])
            multiplied = Multiply()([z_spatial, multi])
            if idx == 0:
                hidden_layer = multiplied
            else:
                hidden_layer = tf.concat([hidden_layer, multiplied], 2)
        hidden_layer = tf.concat([hidden_layer, z_temporal], 2)
    else:
        hidden_layer = z

    if intermediate_dim == 0:
        y = hidden_layer
    else:
        y = Dense(intermediate_dim, activation=tf.nn.tanh, name='dense_3')(hidden_layer)
        # y = tf.keras.layers.BatchNormalization()(y)

    x_decoded = Dense(wspa, activation=tf.nn.tanh, name='dense_4')(y)
    my_loss = Cust_loss()([x, lbl, l, x_decoded, z_shared],
                          loss_weight_share=loss_weight_share, loss_weight_CE=loss_weight_CE)

    pae = Model(inputs=[x, lbl], outputs=[x_decoded, my_loss])

    encoder = Model(x, z)
    pae.summary()
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=1000,
    #     decay_rate=0.9)
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
    pae.compile(optimizer=optimizer)

    return pae, encoder


def prepare_input_paes(windows, labels, nr_ae):
    """
    Prepares input for create_parallel_ae

    Args:
        windows: list of windows
        nr_ae: number of parallel AEs (K in paper)

    Returns:
        array with shape (nr_ae, (nr. of windows)-K+1, window size)
    """
    new_windows = []
    new_labels = []
    nr_windows = windows.shape[0]
    for i in range(nr_ae):
        new_windows.append(windows[i:nr_windows - nr_ae + 1 + i])
        new_labels.append(labels[i:nr_windows - nr_ae + 1 + i])
    return np.transpose(new_windows, (1, 0, 2)), np.transpose(new_labels, (1, 0, 2))


def train_AE(windows, labels, intermediate_dim=0, latent_dim=1, nr_spatial=1, nr_temporal=1,
             nr_ae=3, loss_weight_share=1, loss_weight_CE=1, nr_epochs=200, nr_patience=7):
    """
    Creates and trains an autoencoder with a Time-Invariant REpresentation (TIRE)

    Args:
        windows: time series windows (i.e. {y_t}_t or {z_t}_t in the notation of the paper)
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_shared: number of shared features (should be <= latent_dim)
        nr_ae: number of parallel AEs (K in paper)
        loss_weight: lambda in paper
        nr_epochs: number of epochs for training
        nr_patience: patience for early stopping

    Returns:
        returns the TIRE encoded windows for all windows
    """
    window_size_per_ae = windows.shape[-1]
    new_labels = tf.keras.utils.to_categorical(labels)
    label_size_per_ae = new_labels.shape[-1]
    new_windows, new_labels = prepare_input_paes(windows, new_labels , nr_ae)

    pae, encoder = create_parallel_aes(window_size_per_ae, label_size_per_ae, labels, intermediate_dim,
                                       latent_dim, nr_ae, nr_spatial, nr_temporal, loss_weight_share, loss_weight_CE)
    # nr_classes = np.unique(labels).size
    # if nr_classes != windows.shape[0]:
    #     pae.load_weights('model_weights.h5', by_name=True, skip_mismatch=True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=nr_patience)

    pae.fit({'data': new_windows, 'lbl': new_labels},
            epochs=nr_epochs,
            verbose=1,
            batch_size=32,
            shuffle=True,
            validation_split=0.0,
            initial_epoch=0,
            callbacks=[callback]
            )
    # pae.save_weights('model_weights.h5')
    # reconstruct = pae.predict(new_windows)
    encoded_windows_pae = encoder.predict(new_windows)
    encoded_windows = np.concatenate((encoded_windows_pae[:, 0, :], encoded_windows_pae[-nr_ae + 1:, nr_ae - 1, :]),
                                     axis=0)
    encoded_windows_shared = np.concatenate((encoded_windows_pae[:, 0, :nr_spatial+nr_temporal],
                                      encoded_windows_pae[-nr_ae + 1:, nr_ae - 1, :nr_spatial+nr_temporal]), axis=0)
    return encoded_windows_shared, encoded_windows


