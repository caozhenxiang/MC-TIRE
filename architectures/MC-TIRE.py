import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
if os.environ["CUDA_VISIBLE_DEVICES"] == "0":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Layer, Conv1D, Conv1DTranspose, Flatten
from tensorflow.keras.models import Model
import numpy as np

####################################################
# MC-TIRE model
####################################################
class Cust_loss(Layer):
    def __init__(self, **kwargs):
        super(Cust_loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # split inputs
        x, x_decoded, z_shared_AS, z_shared_B, cov_mat = inputs

        # define losses
        # reconstruction loss
        squared_diff = K.square(x - x_decoded)
        mse_loss = tf.reduce_mean(squared_diff)

        # shared loss A
        shared_AS = tf.reduce_mean(tf.abs(z_shared_AS[:, 1:, :] - z_shared_AS[:, :1, :]))
        shared_B = tf.reduce_mean(tf.abs(z_shared_B[:, 1:, :] - z_shared_B[:, :1, :]))
        shared_loss_AS = kwargs['loss_weight_share_AS'] * shared_AS
        shared_loss_B = kwargs['loss_weight_share_B'] * shared_B
        if tf.math.is_nan(shared_loss_AS):
            shared_loss_AS = 0.0
        if tf.math.is_nan(shared_loss_B):
            shared_loss_B = 0.0

        cov1 = tf.abs(cov_mat[0, :, :])
        cov2 = tf.abs(cov_mat[1, :, :])
        id_mat = tf.eye(cov_mat.shape[1])
        uncor_loss = kwargs['loss_weight_uncor'] * tf.reduce_mean(tf.abs(cov1 - id_mat) + tf.abs(cov2 - id_mat))

        # add losses to model
        self.add_loss(mse_loss, inputs=True)
        self.add_metric(mse_loss, aggregation="mean", name="mse_loss")
        self.add_loss(shared_loss_AS, inputs=True)
        self.add_metric(shared_loss_AS, aggregation="mean", name="shared_loss_AS")
        self.add_loss(shared_loss_B, inputs=True)
        self.add_metric(shared_loss_B, aggregation="mean", name="shared_loss_B")
        self.add_loss(uncor_loss, inputs=True)
        self.add_metric(uncor_loss, aggregation="mean", name="uncorrelated_loss_B")

        total_loss = mse_loss + shared_loss_AS + shared_loss_B + uncor_loss
        return total_loss


def create_parallel_AEs(X, n_filter, enable_summary, loss_weight_share_AS, loss_weight_share_B, loss_weight_uncor, rank):
    initializer = tf.keras.initializers.GlorotUniform()
    initializer_cnn = tf.keras.initializers.GlorotUniform()

    # encoder
    input = Input(shape=(X.shape[1], X.shape[2], X.shape[3]), name="data")
    x1 = input[:, 0, :, :]
    x2 = input[:, 1, :, :]
    nr_channels = X.shape[3]

    B0 = Flatten()(x1)
    B1 = Flatten()(x2)
    desired_B = tf.concat([tf.expand_dims(B0, axis=1), tf.expand_dims(B1, axis=1)], axis=1)
    for idx in np.arange(0, X.shape[3]):
        # desired_B = input[:,:,:,idx]
        enc_wc_l1 = Dense(20, kernel_initializer=initializer, activation='relu')
        enc_wc_l2 = Dense(n_filter, kernel_initializer=initializer, activation='tanh')
        enc_wc_l3 = Dense(1, kernel_initializer=initializer, activation='tanh')
        dec_wc_l1 = Dense(20, kernel_initializer=initializer, activation='relu')
        dec_wc_l2 = Dense(X.shape[2], kernel_initializer=initializer, activation='tanh')
        B_enc = enc_wc_l1(desired_B)
        B_shared = enc_wc_l2(B_enc)
        B_unshared = enc_wc_l3(B_enc)
        B = tf.concat([B_shared, B_unshared], -1)
        B_dec = dec_wc_l1(B)
        B_decoded = dec_wc_l2(B_dec)
        if idx == 0:
            B_decoded_all = tf.expand_dims(B_decoded, axis=-1)
            B_shared_all = B_shared
        else:
            B_decoded_all = tf.concat((B_decoded_all, tf.expand_dims(B_decoded, axis=-1)), axis=-1)
            B_shared_all = tf.concat((B_shared_all, B_shared), axis=-1)

    # Mixing matrix A
    for i in np.arange(rank):
        conv_A_L1 = Conv1D(filters=8, kernel_size=7, padding="same", kernel_initializer=initializer_cnn, strides=2,
                           activation="relu")
        conv_A_L2 = Conv1D(filters=16, kernel_size=7, padding="same", kernel_initializer=initializer_cnn, strides=2,
                           activation="relu")
        conv_A_L3 = Conv1D(filters=nr_channels, kernel_size=7, padding="same", kernel_initializer=initializer_cnn,
                           strides=2, activation="tanh")
        conv_A1 = conv_A_L1(x1)
        conv_A1 = conv_A_L2(conv_A1)
        conv_A1 = conv_A_L3(conv_A1)

        conv_A2 = conv_A_L1(x2)
        conv_A2 = conv_A_L2(conv_A2)
        conv_A2 = conv_A_L3(conv_A2)
        a = tf.concat([tf.expand_dims(tf.reduce_mean(conv_A1, axis=1, keepdims=True), axis=1),
                       tf.expand_dims(tf.reduce_mean(conv_A2, axis=1, keepdims=True), axis=1)], axis=1)
        if i == 0:
            A = a
        else:
            A = tf.concat([A, a], axis=2)

    # Sources S
    for i in np.arange(rank):
        conv_S_L1 = Conv1D(filters=4, kernel_size=7, padding="same", kernel_initializer=initializer_cnn, strides=2,
                           activation="tanh")
        deconv_S_L1 = Conv1DTranspose(filters=4, kernel_size=7, padding="same", kernel_initializer=initializer_cnn,
                                      strides=2, activation="relu")
        deconv_S_L2 = Conv1DTranspose(filters=1, kernel_size=7, padding="same", kernel_initializer=initializer_cnn,
                                      strides=1, activation="tanh")

        conv_S1 = conv_S_L1(x1)
        conv_S2 = conv_S_L1(x2)
        encoded_s = tf.expand_dims(tf.concat([tf.reduce_mean(conv_S1, axis=1, keepdims=True),
                                              tf.reduce_mean(conv_S2, axis=1, keepdims=True)], axis=1), axis=-1)
        deconv_S1 = deconv_S_L1(conv_S1)
        deconv_S2 = deconv_S_L1(conv_S2)
        deconv_S1 = deconv_S_L2(deconv_S1)
        deconv_S2 = deconv_S_L2(deconv_S2)
        s = tf.concat([tf.expand_dims(deconv_S1, axis=1), tf.expand_dims(deconv_S2, axis=1)], axis=1)
        if i == 0:
            S = s
            encoded_S = encoded_s[:,:,:-1,:]
        else:
            S = tf.concat([S, s], axis=-1)
            encoded_S = tf.concat([encoded_S, encoded_s[:,:,:-1,:]], axis=-1)

    across_channel_info = tf.matmul(S, A)
    A = tf.reshape(A, [-1, 2, nr_channels * rank])
    encoded_S = tf.reshape(encoded_S, [-1, 2, 3 * rank])

    # addition to get X
    reconstructed_input = across_channel_info + B_decoded_all

    covriance_matrix_B = tfp.stats.covariance(B_decoded_all, sample_axis=[0, 2], event_axis=-1)

    # add classifier
    z_shared_AS = tf.concat([A, encoded_S], axis=-1)
    z_shared_B = B_shared_all

    my_loss = Cust_loss()([input, reconstructed_input, z_shared_AS, z_shared_B, covriance_matrix_B],
                          loss_weight_share_AS=loss_weight_share_AS, loss_weight_share_B=loss_weight_share_B,
                          loss_weight_uncor=loss_weight_uncor)

    pae = Model(inputs=input, outputs=[reconstructed_input, my_loss])
    encoder = Model(input, [z_shared_AS, z_shared_B])
    autoencoder = Model(input, reconstructed_input)
    if enable_summary:
        pae.summary()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    pae.compile(optimizer=optimizer)
    ae_AS = Model(input, across_channel_info)
    ae_B = Model(input, B_decoded_all)

    return pae, encoder, ae_AS, ae_B, autoencoder


def prepare_inputs(windows, nr_ae=2):
    new_windows = []
    nr_windows = windows.shape[0]
    for i in range(nr_ae):
        new_windows.append(windows[i:nr_windows - nr_ae + 1 + i])
    return np.transpose(new_windows, (1, 0, 2, 3))


def train_model(windows, loss_weight_share_AS, loss_weight_share_B, loss_weight_uncor, n_filter,
                verbose, enable_summary, rank, nr_epochs=200, nr_patience=10):
    new_windows = prepare_inputs(windows)
    pae, encoder, ae_AS, ae_B, ae = create_parallel_AEs(new_windows, n_filter, enable_summary, loss_weight_share_AS,
                                                        loss_weight_share_B, loss_weight_uncor, rank)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=nr_patience)

    pae.fit({'data': new_windows},
            epochs=nr_epochs,
            verbose=verbose,
            batch_size=64,
            shuffle=True,
            validation_split=0.0,
            initial_epoch=0,
            callbacks=[callback]
            )

    encoded_windows_AS, encoded_windows_B = encoder.predict(new_windows)
    encoded_windows_AS = np.concatenate((encoded_windows_AS[:, 0, :], encoded_windows_AS[-1:, 1, :]), axis=0)
    encoded_windows_B = np.concatenate((encoded_windows_B[:, 0, :], encoded_windows_B[-1:, 1, :]), axis=0)
    rec_AS = ae_AS.predict(new_windows)
    rec_AS_windows = np.concatenate((rec_AS[:, 0, :, :], np.expand_dims(rec_AS[-1, 1, :, :], axis=0)), axis=0)
    rec_AS_E_var = np.var(np.sum(np.square(rec_AS_windows), axis=1), axis=0)

    rec_B = ae_B.predict(new_windows)
    rec_B_windows = np.concatenate((rec_B[:, 0, :, :], np.expand_dims(rec_B[-1, 1, :, :], axis=0)), axis=0)
    rec_B_E_var = np.var(np.sum(np.square(rec_B_windows), axis=1), axis=0)

    return encoded_windows_AS, encoded_windows_B, rec_AS_E_var, rec_B_E_var