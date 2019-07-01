from trainers.trainer_abstract import AbstractTrainer
import numpy as np
from data.loader import loader as load
from settings import settings
cfg = settings()


class NONGANTrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network
        return None

    def get_model(self):
        return self.network

    def set_model_weights(self, path):
        self.network.model.load(path)

    def train(self, data=None, labels=None, **kwargs):
        dof = cfg.simulation.dof

        data = load.training_data
        labels = load.training_labels
        train_data = [
            labels[:, :2*dof],  # inital states
            labels[:, 2*dof:4*dof],  # final states
            data[:, :-2],  # initial costate
            data[:, -1],  # final time
            data[:, -2],  # final cost
        ]
        self.network.model.train(train_data)
