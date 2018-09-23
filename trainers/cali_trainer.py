from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.loader import loader as load
from trainers.trainer_abstract import AbstractTrainer


class CALITrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network
        return None

    def get_model(self):
        return self.network

    def set_model_weights(self, path):
        self.network.restore_weights(path)

    def train(self,
              data,
              labels,
              data_test=None,
              labels_test=None,
              epochs=10000,
              batch_size=32,
              d_steps=3,
              verbose=1):

        # load.set_training_data(data)
        # load.set_training_labels(labels)

        print("Training CALI")
        start = timer()
        for it in tqdm(range(epochs)):
            for _ in range(d_steps):
                X_mb, y_mb = load.next_batch(batch_size)
                D_loss = self.network.train_discriminator(X_mb, y_mb)

            # X_mb, y_mb = load.next_batch(batch_size)
            G_loss = self.network.train_generator(X_mb, y_mb)
            self.network.log(X_mb, y_mb, it)

            if it % 1000 == 0:
                n = 1000
                X_mb, y_mb = load.next_batch(n)
                samples = self.network.predict(y_mb)

                n_dof = int(y_mb.shape[1]/4)

                if n_dof == 1:
                    self.plot_unit_circle(X_mb, samples)

                # plt.figure(figsize=(10, 7))
                # for dof in range(1, n_dof*2+1):
                #     self.plot_hist_costates(dof, n_dof, X_mb, samples)
                # plt.savefig('./tmp/out/{}_costates.png'.format(str(0).zfill(3)), bbox_inches='tight')

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
