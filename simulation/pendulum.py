"""
Pendulum class is a simulation implementation. This class generates the necessary data
by means of numerical integration and indirect optimal control. The data is then used
in the offline learning algorithm in order to learn a function approximation for the 
online planning.

All methods required to be exposed are public. The rest is private.

@author: Nick Tsutsunava 2018
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from simulation.pendulumDataGenerator import *


class Pendulum():
    def __init__(self, mode="time", dof=1, u_max=False, **kwargs):
        self.training_data_file = None
        self.training_data_dir = None
        self.mode = mode

        self.dof = dof
        # params
        self.alpha = 1.0
        self.alpha_list = []
        self.u_max = 2 / np.pi - 0.1 if not u_max else u_max
        self.time_weight = 1.0

        self.sampling_bounds = [[-3/2*np.pi, np.pi], [-np.pi, np.pi]]
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
        self.data, self.labels = generateByMode(self.mode, samples, dt)
        return self.data, self.labels

    def simulate_steer_full(self, s0, u, dt=0.01):
        tf = u[2]
        state_action = np.zeros((4,))
        state_action[:2] = s0
        state_action[2:] = u[:2]

        return RK4simulate(self.get_eom(), state_action, dt, tf, self.alpha)

    def simulate_steer(self, s0, u, dt=0.01):
        _, s1, s = self.simulate_steer_full(s0, u, dt=0.01)

        return s1[:2], s[:,:2]

    def get_eom(self):
        if self.mode == "time":
            return lambda s: pendulumSwingUpEOM(s, self.u_max)
        if self.mode == "bounded":
            return lambda s: boundedEnergyPendulumEOM(s, self.u_max, self.time_weight)

    def get_alpha_sampler(self):
        if self.mode == "time":
            return lambda th, om, lh, mh: computeAlphaSwingUp(th, om, lh, mh, self.u_max)
        if self.mode == "bounded":
            return lambda th, om, lh, mh: computeAlphaBoundedEnergy(th, om, lh, mh, self.u_max, self.time_weight)

    def compute_alpha(self, s0, u, default=1.0):
        sample_alpha = self.get_alpha_sampler()
        try:
            self.alpha = sample_alpha(s0[0], s0[1], u[0], u[1])
        except NegativeAlphaError:
            self.alpha = default
        return self.alpha

    def validate_costates(self, s0, u):
        return self.compute_alpha(s0, u, default=False)

    def select_min_alpha(self, idx):
        self.alpha = self.alpha_list[idx]

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
        filename = os.path.join(path, "pendulum_{}.csv".format(self.mode))
        np.savetxt(
            filename,
            np.hstack([self.data, self.labels]),
            delimiter=',',
            comments="",
            fmt="%f",
            header='lambda0, mu0, t_f, cost, theta0, omega0, theta1, omega1')
        return None

    def load(self, path):

        return None
