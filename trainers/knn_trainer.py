from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from tqdm import tqdm

from trainers.trainer_abstract import AbstractTrainer

from data.loader import loader as load

class KNNTrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network

        return None

    def get_model(self):
        return self.network

    def set_model_weights(self, path):
        self.network.model = joblib.load(path)
        pass

    def train(self,
              data=None,
              labels=None):

        print("Fitting kNN model")
        labels = load.training_labels
        data = load.training_data
        self.network.model.fit(labels, data)
