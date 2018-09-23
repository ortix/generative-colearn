from trainers.trainer_abstract import AbstractTrainer
import numpy as np
from data.loader import loader

class CVAETrainer(AbstractTrainer):
    def __init__(self, network):

        self.network = network
        return None

    def get_model(self):
        return self.network

    def set_model_weights(self, weights):
        self.network.decoder.load_weights(weights)
        return None

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

        # Compiling the network results in a model (just a naming convention)
        model = self.network.compile()
        model.summary()

        if data is not None:
            loader.set_training_data(data)
        if labels is not None:
            loader.set_training_labels(labels)

        model.fit(
            [data, labels],
            data,
            shuffle=shuffle,
            epochs=epochs,
            batch_size=batch_size,
            # validation_split=validation_split,
            verbose=verbose,
            callbacks=self.network.callbacks)

        return None
