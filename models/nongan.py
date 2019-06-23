from models.nongenerativemodel import NonGenerativeModel
import numpy as np


class NONGAN:
    def __init__(self):
        config = {
            "dof": 3,
            "layerSize": 256,
            "numLayers": 3,
            "maxTime": 1.0,
            "maxCost": 1.0,
            "maxStates": [1.6, 0.8, 0.8, 1.0, 1.0, 1.0],
            "epochs": 1,
            "batch_size": 2048,
        }
        self.model = NonGenerativeModel(config)

    def predict(self, labels):
        initial_state, final_state = np.split(labels, 2, axis=1)
        costate, time, cost, reachable = self.model.predict(initial_state, final_state)

        return np.hstack([costate, time, cost])

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, path):
        return self.model.load(path)
