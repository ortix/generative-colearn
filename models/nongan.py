from models.nongenerativemodel import NonGenerativeModel
from settings import settings
import numpy as np

cfg = settings()


class NONGAN:
    def __init__(self, **kwargs):
        kwargs["dof"] = cfg.simulation.dof
        self.model = NonGenerativeModel(kwargs)

    def predict(self, labels):
        initial_state, final_state = np.split(labels, 2, axis=1)
        costate, time, cost, reachable = self.model.predict(initial_state, final_state)
        return np.hstack([costate, time, cost])

    def discriminate(self, labels):
        initial_state, final_state = np.split(labels, 2, axis=1)
        _, _, _, reachable = self.model.predict(initial_state, final_state)
        return reachable

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, path):
        return self.model.load(path)