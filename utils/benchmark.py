'''
Benchmarking class to test DGMs on various datasets.
Future improvement is to test DGM conditionally and generatively
'''

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
# Module imports
from munch import munchify

from utils.helpers import *
from settings import settings


class Benchmark():

    def __init__(self, model_name):
        self.cfg = settings()
        self.model_name = model_name
        self.structure = self.get_structure(self.model_name)

        self.model = get_model(self.model_name)(**self.structure)
        self.trainer = get_trainer(self.model_name)(self.model)

        return None

    def generate_sample(self, label):
        trained_model = self.trained_model
        z_dim = self.structure["latent_size"]
        l_dim = self.structure["label_size"]

        noise_label = np.zeros((1, z_dim+l_dim,))
        # noise_label[0, :z_dim] = np.random.normal(0., 1., size=z_dim)
        # noise_label[0, z_dim:] = to_categorical(label)

        im = trained_model.predict(np.eye(10))

        plt.imshow(im.reshape(28, 28))
        plt.show()
        return None

    def generate_samples(self):
        trained_model = self.trainer.get_model()
        z_dim = self.structure["latent_size"]
        l_dim = self.structure["label_size"]
        r, c = 2, 5
        noise = np.random.normal(-1, 1, (r * c, z_dim))
        labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = trained_model.predict(np.eye(10))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28))
                axs[i, j].set_title("Digit: %d" % labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        return None

    def load_model(self):
        try:
            self.trainer.set_model_weights(self.get_weights_filename())
        except Exception as e:
            print(repr(e))
            return False
        return True

    def save_model(self):
        self.trained_model.save_weights(self.get_weights_filename())

    def get_weights_filename(self):
        model_file = os.path.join(self.cfg.paths.models, self.model_name)
        return "{}_benchmark_weights.h5".format(model_file)

    def train_model(self):
        x_data, y_data=self.get_dataset()
        self.trainer.train(x_data, y_data, epochs=self.epochs)
        self.trained_model=self.trainer.get_model()
        self.save_model()
        return None

    def get_dataset(self):
        (x_train, y_train), (x_test, y_test)=mnist.load_data()

        y_train=to_categorical(y_train)
        y_test=to_categorical(y_test)
        x_train=(x_train.astype('float32') - 127.5) / 127.5
        x_test=x_test.astype('float32') / 255.
        x_train=x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test=x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        return x_train, y_train

    def get_structure(self, model):
        if model == "clsgan":
            self.epochs = 10000
            return {
                "input_size": 784,
                "label_size": 10,
                "latent_size": 64,
                "layers": {"g": [
                    128,
                    256
                ],
                    "d": [512, 512]}
            }

        # default to cvae
        self.epochs = 10
        return {
            "input_size": 784,
            "label_size": 10,
            "latent_size": 2,
            "batch_norm": True,
            "layers": [
                128,
                64
            ]
        }


if __name__ == '__main__':
    b=Benchmark('clsgan')
    if b.load_model():
        b.train_model()
    b.generate_samples()
    pass
