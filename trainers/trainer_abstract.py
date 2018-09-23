from abc import ABC, abstractmethod


class AbstractTrainer(ABC):

    @abstractmethod
    def __init__(self, network):
        self.network = network
        return None

    @abstractmethod
    def get_model(self):
        # Needs to be renamed to get_trained_model
        pass

    @abstractmethod
    def train(self,
              data=None,
              labels=None,
              data_test=None,
              labels_test=None,
              epochs=1000,
              batch_size=100,
              shuffle=True,
              validation_split=0.2,
              verbose=1):
        pass
