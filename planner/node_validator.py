import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import time


class NodeValidator():

    def __init__(self, X, d_max, gan_model=None):
        self.d_max = d_max
        self.gan_model = gan_model
        self.reachability_log = str(d_max) if gan_model is None else "Discriminator"
        # Use clean dataset
        print("Fitting node validator")
        self.nb = NN(n_neighbors=2).fit(X)
        return None

    def validate(self, node_list, random_state):
        """
        Return the indices of nodes that are valid to expand from. Only trajectories
        that have been seen during training (or similar) are valid.
        """
        tiled = np.tile(random_state, (node_list.shape[0], 1))
        if self.gan_model is not None:
            val = self.gan_model.discriminate(np.hstack([node_list, tiled]))
            valid_idx = np.where(val.flatten() > 0)[0]
        else:
            distances, idx = self.nb.kneighbors(np.hstack([node_list, tiled]), 1)
            valid_idx = np.flatnonzero(distances < self.d_max)
        return valid_idx
