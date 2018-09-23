import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trainers.trainer_abstract import AbstractTrainer


class CWGANTrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network
        self.callback = None
        return None

    def get_model(self):

        return self.network.generator

    def set_callback(self, callback):
        self.callback = callback
        return None

    def _plot_results(self, d_loss, g_loss):
        ax = pd.DataFrame(
            {
                'Generative Loss': g_loss[:, 0],
                'Discriminative Loss': d_loss[:, 0],
            }
        ).plot(title='Training loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        return None

    def train(self,
              data,
              labels,
              data_test=None,
              labels_test=None,
              epochs=1000,
              batch_size=32,
              training_ratio=5,
              shuffle=True,
              validation_split=0.2,
              verbose=1):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty

        d_loss = np.zeros((epochs, 4))
        g_loss = np.zeros((epochs, 1))
        for epoch in range(epochs):

            # Train discriminator
            for _ in range(training_ratio):
                idx = np.random.randint(0, data.shape[0], batch_size)
                labels_batch = labels[idx]
                data_batch = data[idx]
                noise = np.random.normal(-1, 1, (batch_size, self.network.latent_size))
 
                d_loss[epoch, :] = self.network.d_model.train_on_batch(
                    [data_batch, noise, labels_batch], [valid, fake, dummy])

            # Train generator
            idx = np.random.randint(0, data.shape[0], batch_size)
            labels_batch = labels[idx]
            noise = np.random.normal(-1, 1, (batch_size, self.network.latent_size))
            g_loss[epoch] = self.network.g_model.train_on_batch([noise, labels_batch], valid)
            # exit()
            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[epoch,0], g_loss[epoch]))

        self._plot_results(d_loss, g_loss)

        return None
