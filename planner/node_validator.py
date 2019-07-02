import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from models.nongan import NONGAN
from models.clsgan import CLSGAN
import time


class NodeValidator:
    def __init__(self, X, d_max, gan_model=None):
        self.d_max = d_max
        self.gan_model = gan_model
        self.reachability_log = (
            str(d_max) if not self._is_neural_network(gan_model) else "Discriminator"
        )
        # Use clean dataset
        print("Fitting node validator")
        self.nb = NN(n_neighbors=2).fit(X)
        return None

    def validate(self, node_list, random_state):
        """
        Return the indices of nodes that are valid to expand from. Only trajectories
        that have been seen during training (or similar) are valid.
        """
        if self._is_neural_network(self.gan_model):
            return self._validateNeural(node_list, random_state)
        else:
            return self._validateKNN(node_list, random_state)

    def _is_neural_network(self, model):
        return isinstance(model, NONGAN) or isinstance(model, CLSGAN)

    def _validateKNN(self, node_list, random_state):
        tiled = np.tile(random_state, (node_list.shape[0], 1))
        distances, idx = self.nb.kneighbors(np.hstack([node_list, tiled]), 1)
        valid_idx = np.flatnonzero(distances < self.d_max)
        return valid_idx

    def _validateNeural(self, node_list, random_state):
        tiled = np.tile(random_state, (node_list.shape[0], 1))
        val = self.gan_model.discriminate(np.hstack([node_list, tiled]))
        valid_idx = np.where(val.flatten() > 0)[0]
        return valid_idx

    def validation_analysis(self, node_list, random_state, knn_dmax):
        dmax = self.d_max
        self.d_max = knn_dmax
        valid_idx_knn = self._validateKNN(node_list, random_state)
        self.d_max = dmax
        valid_idx_neural = self._validateNeural(node_list, random_state)
        return valid_idx_knn, valid_idx_neural

