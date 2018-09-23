from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers, losses, metrics, regularizers
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard
from keras.layers import (BatchNormalization, Concatenate, Dense, Input,
                          Lambda, PReLU)
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, load_model
from keras.utils import plot_model, to_categorical


class CVAE():
    def __init__(self,
                 input_size,
                 label_size,
                 latent_size,
                 layers,
                 optimizer='rmsprop',
                 show_metrics=False,
                 batch_norm=True):
        # variables
        self.input_size = input_size
        self.label_size = label_size
        self.latent_size = latent_size
        self.layer_sizes = layers
        self.batch_norm = batch_norm
        self.initializer = 'he_uniform'
        self.regu = None
        # Build inputs tensor containing the input data and conditional array
        self.input = Input(shape=(input_size, ), name="input_data")
        self.conditional = Input(shape=(label_size, ), name="input_labels")
        self.inputs = concatenate([self.input, self.conditional])

        # Build encoder and decoder
        self.mu, self.log_sigma = self.create_encoder(self.inputs)
        self.decoder, self.d_log_sigma = self.create_decoder()
        self.sampler = self.decoder([Lambda(self.sample_latent)([self.mu, self.log_sigma]), self.conditional])

        # Generate Keras models for the encoder and the entire VAE
        self.encoder = Model([self.input, self.conditional], self.mu)
        self.model = Model([self.input, self.conditional], self.sampler)
        self.optimizer = optimizer
        self.verbose = show_metrics
        self.callbacks = []

        # Run some post operations
        self.init_callbacks()

    # returns two tensors, one for the encoding (z_mean), one for making the manifold smooth
    def create_encoder(self, nn_input):
        x = nn_input
        for l in self.layer_sizes:
            x = Dense(l, kernel_initializer=self.initializer,
                      kernel_regularizer=self.regu)(x)
            x = PReLU()(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
        z_mu = Dense(self.latent_size, activation="linear", name="z_mean")(x)
        z_log_sigma = Dense(self.latent_size, activation="linear", name="z_log_sigma")(x)
        return z_mu, z_log_sigma

    def create_decoder(self):
        noise = Input(shape=(self.latent_size,))
        label = Input(shape=(self.label_size,))
        x = concatenate([noise, label])
        for l in self.layer_sizes[::-1]:
            x = Dense(l, kernel_initializer=self.initializer,
                      kernel_regularizer=self.regu)(x)
            x = PReLU()(x)
            if self.batch_norm:
                x = BatchNormalization()(x)

        n = int(self.input_size-2)
        costates = Lambda(lambda x: x[:, :2])(x)
        time_cost = Lambda(lambda x: x[:, 2:])(x)
        costates_dense = Dense(n, activation="tanh")(costates)
        time_cost_dense = Dense(n, activation="relu")(time_cost)

        out_mu = Concatenate()([costates_dense, time_cost_dense])
        out_log_sigma = Dense(self.input_size, activation='linear')(x)

        return Model([noise, label], out_mu), out_log_sigma

    # used for training
    def sample_latent(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]),
            mean=0.,
            stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # used for runtime
    def sample_z(self, n):
        return np.random.normal(0.0, 1., size=[n, self.latent_size])

    # loss functions
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.recon_loss(x, x_decoded_mean)
        kl_loss = self.kl_loss(x, x_decoded_mean)
        return -K.mean(xent_loss + 1.0*kl_loss) #0.085

    def recon_loss(self, x, x_decoded_mean):
        return -0.5*losses.mean_squared_error(x, x_decoded_mean)

    def kl_loss(self, x, x_decoded_mean):
        return 0.5 * K.sum(1 + self.log_sigma - K.square(self.mu) - K.exp(self.log_sigma),
                           axis=1)

    # builds and returns the model. This is how you get the model in your training code.
    def compile(self):
        met = []
        # if self.verbose:
        met = [self.recon_loss, self.kl_loss]
        self.model.compile(self.optimizer, loss=self.vae_loss, metrics=met)
        return self.model

    def predict(self, labels):
        noise = self.sample_z(labels.shape[0])
        return self.decoder.predict([noise, labels])

    def load_model(self, path):
        return load_model(path)

    def save_weights(self, path):
        return self.decoder.save_weights(path)

    def init_callbacks(self):
        # We store the runs in subdirectories named by the time
        dirname = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.callbacks.append(TensorBoard(log_dir="./logs/tensorboard/{}/".format(dirname)))

        # self.callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                                         patience=5, min_lr=0.001))

        # self.callbacks.append(LambdaCallback(on_epoch_end=lambda batch, logs: print(logs)))
        return None


if __name__ == "__main__":
    cvae = CVAE(784, 10, 2, [128, 64], optimizer='rmsprop')
