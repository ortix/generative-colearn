import numpy as np


class Node():
    def __init__(self, state):
        '''
        The state argument must be a list of states
        '''
        self.state = np.array(state)
        self.parent = None
        self.trajectory = []

    # def __repr__(self):
    #     return str(self.state)
