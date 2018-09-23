"""
Pendulum class is a simulation implementation. This class generates the necessary data
by means of numerical integration and indirect optimal control. The data is then used
in the offline learning algorithm in order to learn a function approximation for the 
online planning.

All methods required to be exposed are public. The rest is private.

@author: Nick Tsutsunava 2018
"""
import numpy as np
import os
from simulation.pendulumDataGenerator import generator, RK4simulate, pendulumSwingUpEOM
import matplotlib.pyplot as plt


class Pendulum:
    def __init__(self):
        self.training_data_file = None
        self.training_data_dir = None
        return None

    def set_training_data_dir(self, path):
        """ 
        Absolute path to the directory where training data is stored
        """
        self.training_data_dir = path
        if not os.path.exists(path):
            os.makedirs(path)
        return None

    def simulate(self, samples=1000, dt=0.01):
        self.data, self.labels = generator(samples, dt)

        return self.data, self.labels

    def simulate_steer(self, s0, u, dt=0.01):
        tf = u[2]
        state_action = np.zeros((4,))
        state_action[:2] = s0
        state_action[2:] = u[:2]
        u_max = np.pi / 2 - 0.1

        tf_hat, s1 = RK4simulate(lambda s: pendulumSwingUpEOM(s, u_max), state_action, dt, tf)

        return s1[:2]

    def validate(self, s0, s1, u, dt=0.01):
        '''
        Validate by passing in an initial state and the control action
        s0: theta, omega (initial)
        s1: theta, omega (desired/final)
        u: [lambda, mu, tf, cost]
        The function returns the error of the position and velocity separately  
        '''
        s1Hat = self.simulate_steer(s0, u)
        return (np.abs(s1Hat[0] - s1[0]), np.abs(s1Hat[1] - s1[1]))

    def save(self, path):
        filename = os.path.join(path, "pendulum.csv")
        np.savetxt(
            filename,
            np.hstack([self.data, self.labels]),
            delimiter=',',
            header='lambda0, mu0, t1, cost, theta0, omega0, theta1, omega1')
        return None

    def load(self, path):

        return None
