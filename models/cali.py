import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil
from datetime import datetime


def log_tf(x):
    return tf.log(x + 1e-8)


class CALI():

    def __init__(self, input_size, label_size, latent_size, layers, lr=1e-4):
        print("Initializing CALI")
        self.input_size = input_size
        self.label_size = label_size
        self.latent_size = latent_size
        self.layers = layers
        self.X = tf.placeholder(tf.float32, shape=[None, input_size])
        self.y = tf.placeholder(tf.float32, shape=[None, label_size])
        self.z = tf.placeholder(tf.float32, shape=[None, latent_size])

        z_hat = self.Q(self.X, self.y)
        self.X_hat = self.P(self.z, self.y)

        D_enc = self.D(self.X, z_hat, self.y)
        D_gen = self.D(self.X_hat, self.z, self.y, reuse=True)

        self.D_loss = -tf.reduce_mean(log_tf(D_enc) + log_tf(1 - D_gen))
        self.G_loss = -tf.reduce_mean(log_tf(D_gen) + log_tf(1 - D_enc))

        Q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        P_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        theta_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

        self.D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                         .minimize(self.D_loss, var_list=theta_D))
        self.G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
                         .minimize(self.G_loss, var_list=[Q_var, P_var]))

        self._saver = tf.train.Saver()
        self._sess = tf.Session()

        # Tensorboard  init
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        self.merged = tf.summary.merge_all()
        self.logdir = "./logs/tensorboard/{}/".format(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = tf.summary.FileWriter(self.logdir, self._sess.graph)

        self._sess.run(tf.global_variables_initializer())

        return None

    def sample_z(self, m):
        return np.random.uniform(-1, 1., size=[m, self.latent_size])

    def Q(self, X, y):
        inputs = tf.concat(axis=1, values=[X, y])
        layers = self.layers["g"]
        with tf.variable_scope("encoder"):
            dense = tf.layers.dense(inputs, layers[0], activation=tf.nn.leaky_relu)
            for nodes in layers[1:]:
                dense = tf.layers.dense(dense, nodes, activation=tf.nn.leaky_relu)
                dense = tf.layers.batch_normalization(dense)
            return tf.layers.dense(dense, self.latent_size)

    def P(self, z, y):
        inputs = tf.concat(axis=1, values=[z, y])
        layers = self.layers["g"]
        with tf.variable_scope("decoder"):
            dense = tf.layers.dense(inputs, layers[0], activation=tf.nn.leaky_relu)
            for nodes in layers[1:]:
                dense = tf.layers.dense(dense, nodes, activation=tf.nn.leaky_relu)
                dense = tf.layers.batch_normalization(dense)
            return tf.layers.dense(dense, self.input_size, activation=tf.nn.tanh)

    def D(self, X, z, y, reuse=False):
        inputs = tf.concat([X, z, y], axis=1)
        layers = self.layers["d"]
        with tf.variable_scope("discriminator", reuse=reuse):
            dense = tf.layers.dense(inputs, layers[0], activation=tf.nn.relu)
            for nodes in layers[1:]:
                dense = tf.layers.dense(dense, nodes, activation=tf.nn.relu)
                # dense = tf.layers.batch_normalization(dense)
            return tf.layers.dense(dense, 1, activation=tf.nn.sigmoid)

    def train_discriminator(self, X_mb, y_mb):
        z_mb = self.sample_z(X_mb.shape[0])
        _, D_loss_curr = self._sess.run(
            [self.D_solver, self.D_loss], feed_dict={self.X: X_mb, self.z: z_mb, self.y: y_mb}
        )
        return D_loss_curr

    def train_generator(self, X_mb, y_mb):
        z_mb = self.sample_z(X_mb.shape[0])
        _, G_loss_curr = self._sess.run(
            [self.G_solver, self.G_loss], feed_dict={self.X: X_mb, self.z: z_mb, self.y: y_mb}
        )
        return G_loss_curr

    def predict(self, labels):
        z_mb = self.sample_z(labels.shape[0])
        return self._sess.run(self.X_hat, feed_dict={self.z: z_mb, self.y: labels})

    def log(self, X_mb, y_mb, i):
        if i % 50 == 0:
            Z_mb = self.sample_z(X_mb.shape[0])
            summary = self._sess.run(self.merged, feed_dict={self.X: X_mb, self.z: Z_mb, self.y: y_mb})
            self.writer.add_summary(summary, i)
            self.writer.flush()
        return None

    def save_weights(self, path):
        self._saver.save(self._sess, path)

    def restore_weights(self, path):
        self._saver.restore(self._sess, path)
        shutil.rmtree(self.logdir)
