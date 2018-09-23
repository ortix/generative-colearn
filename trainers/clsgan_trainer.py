import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data.loader import loader as load
from trainers.trainer_abstract import AbstractTrainer
from settings import settings

cfg = settings()


class CLSGANTrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network
        return None

    def get_model(self):
        return self.network

    def set_model_weights(self, path):
        self.network.restore_weights(path)

    def train(self,
              data=None,
              labels=None,
              data_test=None,
              labels_test=None,
              epochs=10000,
              batch_size=32,
              d_steps=3,
              verbose=1):

        if data is not None:
            load.set_training_data(data)
        if labels is not None:
            load.set_training_labels(labels)

        print("Training CLSGAN")

        # Push training data into tf.data
        sess = self.network.get_session()
        train_dataset = tf.data.Dataset.from_tensor_slices(load.training_set()).repeat().batch(batch_size)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_init_op = iterator.make_initializer(train_dataset)
        sess.run(train_init_op)
        training_data = iterator.get_next()

        # Start actual training loop
        start = timer()
        loss = np.zeros((epochs, 2))
        for it in tqdm(range(epochs)):
            Z_mb = self.network.sample_z(batch_size)
            for _ in range(d_steps):
                X_mb, y_mb = sess.run(training_data)
                loss[it, 0] = self.network.train_discriminator(Z_mb, X_mb, y_mb)

            # X_mb, y_mb = sess.run(training_data)
            loss[it, 1] = self.network.train_generator(Z_mb, y_mb)
            self.network.log(Z_mb, X_mb, y_mb, it)

            # Set -1 to 0 for plotting
            if it % 1000 == -1:
                n = 1000
                samples = self.network.predict(y_mb)

                n_dof = int(y_mb.shape[1]/4)

                if n_dof == 1:
                    self.plot_unit_circle(X_mb, samples)

                plt.figure(figsize=(10, 7))
                for dof in range(1, n_dof*2+1):
                    self.plot_hist_costates(dof, n_dof, X_mb, samples)
                plt.savefig('./tmp/out/{}_costates.png'.format(str(0).zfill(3)), bbox_inches='tight')

                norm = np.linalg.norm(samples[:, :-2], axis=1)
                plt.figure()
                plt.hist(norm, bins='auto')
                plt.title("Hypersphere proximity of costates")
                plt.xlabel('Norm of points')
                plt.savefig('./tmp/out/{}_hypersphere.png'.format(str(0).zfill(3)), bbox_inches='tight')
                plt.close('all')

        # End training

        elapsed = timer() - start
        print("Training time for {} epochs: {} seconds".format(epochs, elapsed))

        self.plot_training_loss(loss, epochs)
        return None

    def plot_training_loss(self, loss, epochs):
        plt.figure(figsize=(9, 4))
        plt.plot(range(epochs), loss[:, 0], range(epochs), loss[:, 1])
        plt.legend(["Discriminator loss", "Generator loss"])
        plt.ylim(0,0.5)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        self.save_fig("training_loss", subfolder=["out"])

        # Temporary
        path = os.path.join(cfg.paths.tmp, "out", "loss.txt")
        np.savetxt(path, loss)

        return None

    def save_fig(self, name, subfolder=None):
        path = os.path.join(cfg.paths.tmp, *subfolder, "{}.eps".format(name))
        plt.savefig(path, bbox_inches="tight")

    def plot_unit_circle(self, x, samples):
        plt.figure(figsize=(10, 7))
        plt.scatter(x[:, 0], x[:, 1], s=0.5)
        plt.scatter(samples[:, 0], samples[:, 1], s=0.5)
        plt.title("Unit circle divergence")
        plt.xlabel("Lambda")
        plt.ylabel("Mu")
        plt.legend(["True", "Generated"])
        plt.savefig('./tmp/out/unit_circle.png', bbox_inches='tight')
        return None

    def plot_hist_costates(self, n, m, x, samples):
        plt.subplot(2, m, n)
        plt.hist(x[:, n-1], bins='auto')
        plt.hist(samples[:, n-1], bins='auto', alpha=0.7)
        if n > m:
            plt.xlabel("Mu {}".format(n))
        else:
            plt.xlabel("Lambda {}".format(n))
        plt.legend(["True", "Generated"])
