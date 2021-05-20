import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.models import Model
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, windows, labels, batch_size=128, shuffle=True):
        super().__init__()
        self.windows = windows
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.key_array = np.arange(self.windows.shape[0], dtype=np.uint32)

        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index*self.batch_size:(index+1)*self.batch_size]

        x = np.asarray(self.windows[keys], dtype=np.float32)
        y = np.asarray(self.labels[keys], dtype=np.float32)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)


def create_parallel_aes(window_size_per_ae, labels, intermediate_dim=0, latent_dim=1, nr_spatial=1,
                        nr_temporal=1):
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
    x = Input(shape=(wspa,), name='data')
    if intermediate_dim == 0:
        y = x
    else:
        y = Dense(intermediate_dim, activation=tf.nn.tanh, name='dense_1')(x)

    z_spatial = Dense(nr_spatial, activation=tf.nn.tanh, name='dense_2-1')(y)
    z_temporal = Dense(nr_temporal, activation=tf.nn.tanh, name='dense_2-2')(y)
    z_unshared = Dense(nr_unshared, activation=tf.nn.tanh, name='dense_2-3')(y)
    z_shared = tf.concat([z_spatial, z_temporal], -1)
    z = tf.concat([z_shared, z_unshared], -1)
    classes = labels.max() + 1
    l = Dense(classes, activation='softmax', name='dense_5')(z_shared)

    if (nr_spatial != 1) and (nr_spatial != 0) and (nr_unshared != 0):
        splits = tf.split(z_unshared, num_or_size_splits=nr_unshared, axis=1)
        for idx, split in enumerate(splits):
            multi = tf.tile(split, multiples=[1, nr_spatial])
            multiplied = Multiply()([z_spatial, multi])
            if idx == 0:
                hidden_layer = multiplied
            else:
                hidden_layer = tf.concat([hidden_layer, multiplied], 1)
        hidden_layer = tf.concat([hidden_layer, z_temporal], 1)
    else:
        hidden_layer = z

    if intermediate_dim == 0:
        y = hidden_layer
    else:
        y = Dense(intermediate_dim, activation=tf.nn.tanh, name='dense_3')(hidden_layer)
    x_decoded = Dense(wspa, activation=tf.nn.tanh, name='dense_4')(y)

    pae = Model(x, x_decoded)
    encoder = Model(x, z)
    classifier = Model(x, l)

    return pae, encoder, classifier


def train_AE(windows, labels, intermediate_dim=0, latent_dim=1, nr_spatial=1, nr_temporal=1, loss_weight_CE=1,
             loss_weight_center=1, nr_epochs=200, batch_size=32, alpha=0.5):
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
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss_train = np.zeros(shape=(nr_epochs,), dtype=np.float32)
    loss_ce = np.zeros(shape=(nr_epochs,), dtype=np.float32)
    loss_mse = np.zeros(shape=(nr_epochs,), dtype=np.float32)
    loss_center = np.zeros(shape=(nr_epochs,), dtype=np.float32)

    window_size_per_ae = windows.shape[-1]
    encoded_labels = tf.keras.utils.to_categorical(labels)

    generator = DataGenerator(windows=windows, labels=encoded_labels, batch_size=batch_size, shuffle=True)
    n_batches = len(generator)

    pae, encoder, classifier = create_parallel_aes(window_size_per_ae, labels, intermediate_dim, latent_dim, nr_spatial,
                                                   nr_temporal)
    pae.summary()
    encoder.summary()
    classifier.summary()
    nr_shared = nr_spatial+nr_temporal
    centers = tf.random.uniform(shape=(np.unique(labels).size, nr_shared))

    for epoch in range(nr_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()  # Keeping track of the training loss
        epoch_loss_ce = tf.keras.metrics.Mean()  # Keeping track of the training CrossEntropy loss
        epoch_loss_mse = tf.keras.metrics.Mean()  # Keeping track of the training reconstruction loss
        epoch_loss_center = tf.keras.metrics.Mean()  # Keeping track of the training center loss
        print('==== Epoch {}/{} ===='.format(epoch+1, nr_epochs))

        for batch in range(n_batches):
            x, y = generator.__getitem__(batch)

            with tf.GradientTape() as tape:  # Forward pass
                # compute losses
                y_ = classifier(x, training=True)
                ce_loss = loss_weight_CE * cce(y_true=y, y_pred=y_)
                x_decoded = pae(x, training=True)
                mse_loss = tf.reduce_mean(tf.square(x_decoded - x))
                if loss_weight_center == 0:
                    center_loss = 0.0
                else:
                    features = encoder(x, training=True)[:,:nr_shared]
                    center_loss = loss_weight_center * tf.reduce_mean(tf.square(tf.matmul(y, centers) - features))
                    # update centers
                    nr_samples_in_batch = tf.reduce_sum(y, axis=0, keepdims=True)
                    embedding_centers = tf.multiply(centers, tf.transpose(tf.tile(nr_samples_in_batch, [nr_shared, 1])))
                    nr_samples_in_batch = nr_samples_in_batch + tf.ones_like(nr_samples_in_batch)
                    delta_centers = tf.math.divide(embedding_centers - tf.matmul(tf.transpose(y), features),
                                                   tf.transpose(tf.tile(nr_samples_in_batch, [nr_shared, 1])))
                    centers = centers - alpha * delta_centers
                losses = ce_loss + mse_loss + center_loss

            # collect trainable parameters
            pae_para = pae.trainable_variables
            classifier_para = classifier.trainable_variables[-2:]
            trainable_para = pae_para + classifier_para
            grad = tape.gradient(losses, trainable_para)  # Backpropagation
            optimizer.apply_gradients(zip(grad, trainable_para))  # Update network weights

            epoch_loss_avg(losses)
            epoch_loss_ce(ce_loss)
            epoch_loss_mse(mse_loss)
            epoch_loss_center(center_loss)

        generator.on_epoch_end()
        # if epoch % 20 == 0:
        # plt.scatter(test_features[:, 0], test_features[:, 1], c=labels, edgecolor="none", s=5, cmap='gist_rainbow')
        #     plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="*", edgecolor="none", s=50)
        #     plt.show()

        loss_train[epoch] = epoch_loss_avg.result()
        loss_ce[epoch] = epoch_loss_ce.result()
        loss_mse[epoch] = epoch_loss_mse.result()
        loss_center[epoch] = epoch_loss_center.result()

        print('---- Training ----')
        print('Loss  =  {0:.4e}'.format(loss_train[epoch]))
        print('CE Loss  =  {0:.4e}'.format(loss_ce[epoch]))
        print('MSE Loss  =  {0:.4e}'.format(loss_mse[epoch]))
        print('Center Loss  =  {0:.4e}'.format(loss_center[epoch]))

    encoded_windows = encoder.predict(windows)
    shared_encoded_windows = encoded_windows[:,:nr_shared]
    return shared_encoded_windows, encoded_windows

