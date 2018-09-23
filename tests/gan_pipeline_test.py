import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class CLSGAN():

    def __init__(self,
                 input_size,
                 label_size,
                 latent_size,
                 layers,
                 lr=1e-3,
                 beta1=0.5,
                 log=True):
        print("Initializing CLSGAN")
        tf.reset_default_graph()

        # Class properties
        self.input_size = input_size
        self.label_size = label_size
        self.latent_size = latent_size
        self.layer_sizes = layers

        # Create placeholders
        self.data = tf.placeholder(tf.float32, shape=[None, self.input_size], name="data")
        self.labels = tf.placeholder(tf.float32, shape=[None, self.label_size], name="labels")
        self.noise = tf.placeholder(tf.float32, shape=[None, self.latent_size], name="noise")

        self.initializer = tf.contrib.layers.xavier_initializer()

        # Build network
        self.G_sample = self.build_generator(self.noise, self.labels)
        D_real = self.build_discriminator(self.data, self.labels)
        D_fake = self.build_discriminator(self.G_sample, self.labels, reuse=True)
        self.D_sample = D_fake
        self.D_sample_real = D_real

        # LSGAN Loss
        self.D_loss = 0.5 * (tf.reduce_mean(tf.square((D_real - 1))) + tf.reduce_mean(tf.square(D_fake)))
        self.G_loss = 0.5 * tf.reduce_mean(tf.square((D_fake - 1)))

        D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        self.D_solver = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1
        ).minimize(self.D_loss, var_list=D_vars)
        self.G_solver = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1
        ).minimize(self.G_loss, var_list=G_vars)

        self._saver = tf.train.Saver()

        self._sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

        # Tensorboard  init
        tf.summary.scalar("D_loss", self.D_loss)
        tf.summary.scalar("G_loss", self.G_loss)
        self.merged = tf.summary.merge_all()
        self.logdir = "./logs/tensorboard/{}/".format(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = tf.summary.FileWriter(self.logdir, self._sess.graph)

        self._sess.run(tf.global_variables_initializer())

        return None

    def get_session(self):
        return self._sess

    def train_discriminator(self, Z_mb, X_mb, y_mb):
        _, loss = self._sess.run(
            [self.D_solver, self.D_loss],
            feed_dict={self.data: X_mb,
                       self.noise: Z_mb,
                       self.labels: y_mb})
        return loss

    def train_generator(self, Z_mb, y_mb):
        _, loss = self._sess.run([self.G_solver, self.G_loss],
                                 feed_dict={self.noise: Z_mb,
                                            self.labels: y_mb})
        return loss

    def build_discriminator(self, x, y, reuse=False):
        dense = tf.concat(axis=1, values=[x, y])
        with tf.variable_scope('discriminator', reuse=reuse, initializer=self.initializer):
            for nodes in self.layer_sizes["d"]:
                dense = tf.layers.dense(dense, nodes, activation=tf.nn.leaky_relu)
                dense = tf.layers.batch_normalization(dense)
                dense = tf.concat(axis=1, values=[dense, y])
                dense = tf.layers.dropout(dense)
            out = tf.layers.dense(dense, 1)
        return out

    def build_generator(self, z, y):
        # y_dense = tf.layers.dense(y, 4)
        layers = self.layer_sizes["g"]
        dense = tf.concat(axis=1, values=[z, y])
        with tf.variable_scope('generator', initializer=self.initializer):
            for nodes in layers:
                dense = tf.layers.dense(dense, nodes, activation=tf.nn.relu)
                # dense = tf.layers.batch_normalization(dense)
            logits = tf.layers.dense(dense, self.input_size)

            # Manually constrain time and cost tensor
            n = int(self.input_size-2)
            costates, time_cost = tf.split(logits, [n, 2], axis=1)
            costates_dense = tf.layers.dense(costates, n, activation=tf.nn.tanh)
            time_cost_dense = tf.layers.dense(time_cost, 2, activation=tf.nn.relu6)

        return tf.concat(values=[costates_dense, time_cost_dense], axis=1)

    def discriminate(self, labels):
        noise = self.sample_z(labels.shape[0])

        return self._sess.run(self.D_sample,
                              feed_dict={self.noise: noise, self.labels: labels})

    def predict(self, labels):
        noise = self.sample_z(labels.shape[0])

        return self._sess.run(self.G_sample,
                              feed_dict={self.noise: noise, self.labels: labels})

    def sample_z(self, n):
        return np.random.uniform(-1., 1, size=[n, self.latent_size])

    def log(self, Z_mb, X_mb, y_mb, i):
        if i % 50 == 0:
            summary = self._sess.run(self.merged, feed_dict={self.data: X_mb, self.noise: Z_mb, self.labels: y_mb})
            self.writer.add_summary(summary, i)
            self.writer.flush()

    def save_weights(self, path):
        self._saver.save(self._sess, path)

    def restore_weights(self, path):
        self._saver.restore(self._sess, path)
        shutil.rmtree(self.logdir)

    def close_session(self):
        self._sess.close()


if __name__ == "__main__":
    from data.loader import loader

    gan = CLSGAN(6, 8, 32, {"g": [256, 256], "d": [256, 256]})

    loader.init_load("2dof_time.csv")
    batch_size = 100
    train_dataset = tf.data.Dataset.from_tensor_slices(loader.training_set()).repeat().batch(batch_size)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    with gan.get_session() as sess:
        sess.run(train_init_op)
        training_data = iterator.get_next()
        for it in range(30000):
            X_mb, y_mb = sess.run(training_data)
            Z_mb = gan.sample_z(batch_size)
            for _ in range(1):
                D_loss = gan.train_discriminator(Z_mb, X_mb, y_mb)

            # X_mb, y_mb = load.next_batch(batch_size)
            G_loss = gan.train_generator(Z_mb, y_mb)
            # gan.log(Z_mb, X_mb, y_mb, it)

            if it % 50 == 0:
                print("D_loss: {} \t G_loss: {}".format(D_loss, G_loss))
