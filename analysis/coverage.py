import os

import numpy as np

import matplotlib.pyplot as plt

from data.loader import loader as load
from models.clsgan import CLSGAN
from settings import settings
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from scipy.stats import norm as _norm


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class Coverage():
    def __init__(self):
        self.cfg = settings()
        # self.cfg.model.clsgan.structure.label_size = 4
        # self.cfg.model.clsgan.structure.input_size = 4
        # self.model = CLSGAN(**self.cfg.model.clsgan.structure)
        # model_weights = os.path.join(self.cfg.paths.models, "clsgan_2dof_time_dirty_weights.h5")
        # self.model.restore_weights(model_weights)

    def run(self):
        self.eef_pos_1dof()
        self.eef_pos_2dof()
        self.eef_velocity_1dof()
        self.eef_velocity_2dof()
        plt.show()
        return None

    def eef_velocity_2dof(self):
        plt.figure(figsize=(10, 6))
        load.init_load("2dof_time.csv", force=True)
        _, labels = load.training_set()
        n = 8000
        dof = 2
        state_f = labels[:, 2*dof:]
        knn = NearestNeighbors(n_neighbors=n).fit(state_f)
        d, v_idx = knn.kneighbors([[0.25*np.pi, 0*np.pi, 0.0, 0.0]])

        # idx = v_idx
        idx = v_idx.reshape(-1)[np.flatnonzero(d.reshape(-1) < 0.30)]

        nn = state_f[idx.reshape(-1), :]
        plt.subplot(1, 2, 1)
        x1 = np.cos(nn[:, 0]) + np.cos(nn[:, 1])
        y1 = 1.0*(np.sin(nn[:, 0]) + np.sin(nn[:, 1]))
        omega = np.linalg.norm(nn[:, dof:], axis=1)
        plt.scatter(x1, y1, s=1, alpha=0.5, c=omega)
        plt.colorbar().set_label('Norm of omega')
        plt.axis('equal')
        plt.title('{} nearest neighbors to goal state'.format(idx.shape[0]))
        plt.xlabel("x")
        plt.ylabel("y")

        plt.subplot(1, 2, 2)
        plt.hist(d.reshape(-1), bins='auto')
        plt.title('Euclidean distance to goal')
        plt.xlabel('d')

        plt.show()

    def eef_velocity_1dof(self):
        plt.figure(figsize=(10, 6))
        load.init_load("pendulum_time.csv", force=True)
        _, labels = load.training_set()

        dof = 1
        state_f = labels[:, 2*dof:]
        n = 1000
        knn = NearestNeighbors(n_neighbors=1000).fit(state_f[:, :2*dof])
        d, v_idx = knn.kneighbors([[0, 0]])

        idx = v_idx.reshape(-1)[np.flatnonzero(d.reshape(-1) < 0.15)]
        # idx= v_idx

        nn = state_f[idx.reshape(-1), :]

        plt.subplot(1, 2, 1)
        x1 = np.sin(nn[:, 0])  # + np.cos(nn[:, 1])
        y1 = 1.0*(np.cos(nn[:, 0]))  # + np.sin(nn[:, 1]))
        omega = np.linalg.norm(nn[:, dof:], axis=1)
        plt.scatter(x1, y1, s=1, alpha=0.5, c=omega)
        plt.colorbar().set_label('Norm of omega')
        plt.axis('equal')
        plt.title('Neighbors within 0.15 to goal state')
        plt.xlabel("x")
        plt.ylabel("y")

        plt.subplot(1, 2, 2)
        plt.hist(d.reshape(-1), bins='auto')
        plt.title('Euclidean distance to goal')
        plt.xlabel('d')

        return None

    def eef_pos_1dof(self):
        """
        Display k=10000 configurations of the end effector
        """
        plt.figure(figsize=(10, 6))
        load.init_load("pendulum_time.csv", force=True)
        _, labels = load.training_set()

        k = 10000
        idx = np.random.randint(0, labels.shape[0], k)

        state_i = labels[idx, :2]
        state_f = labels[idx, 2:]

        plt.subplot(1, 2, 1)
        # Initial positions of the arm
        x1, y1 = pol2cart(1, state_i[:, 0])
        omega = state_i[:, 1]
        plt.scatter(x1, y1, s=2, alpha=0.8, c=omega)
        plt.colorbar().set_label('Norm of omega')
        plt.axis('equal')
        plt.title('Initial position end effector')

        # Down down position
        x, y = pol2cart(1, -0.5*np.pi)
        plt.plot([0, x], [0, y], '-k', linewidth=2)

        # Final positions of the arm
        plt.subplot(1, 2, 2)
        x2, y2 = pol2cart(1, state_f[:, 0])
        omega = state_f[:, 1]
        plt.scatter(x2, y2, s=2, alpha=0.8, c=omega)
        plt.colorbar().set_label('Norm of omega')
        plt.axis('equal')
        plt.title('Final position end effector')

        # up up position
        x, y = pol2cart(1, 0.5*np.pi)
        plt.plot([0, x], [0, y], '-k', linewidth=2)
        return None

    def eef_pos_2dof(self):
        """
        Display k=5000 configurations of the end effector
        """
        plt.figure(figsize=(10, 6))
        load.init_load("2dof_time.csv", force=True)
        _, labels = load.training_set()

        # idx = np.random.randint(0, labels.shape[0], k)
        idx_a = np.arange(labels.shape[0])
        np.random.shuffle(idx_a)
        k = 5000
        idx = idx_a[:k]

        dof = 2
        state_i = labels[idx, :2*dof]
        state_f = labels[idx, 2*dof:]

        plt.subplot(1, 2, 1)
        # Initial positions of the arm
        x1 = np.cos(state_i[:, 0]) + np.cos(state_i[:, 1])
        y1 = 1.0*(np.sin(state_i[:, 0]) + np.sin(state_i[:, 1]))
        omega = np.linalg.norm(state_i[:, 2:], axis=1)
        c = np.linalg.norm(np.tile([-0.25*np.pi, 0, 0, 0], (state_i.shape[0], 1))-state_i, axis=1)
        plt.scatter(x1, y1, s=2, alpha=0.8, c=c)
        plt.colorbar().set_label('Euclidean distance to start pos')
        plt.axis('equal')
        plt.title('Initial position end effector')

        # Start position
        x, y = pol2cart(1, -0.25*np.pi)
        plt.plot([0, x, x+1], [0, y, y], '-r', linewidth=2)

        # Final positions of the arm
        plt.subplot(1, 2, 2)
        x2 = np.cos(state_f[:, 0]) + np.cos(state_f[:, 1])
        y2 = 1.0*(np.sin(state_f[:, 0]) + np.sin(state_f[:, 1]))
        omega = np.linalg.norm(state_f[:, 2:], axis=1)
        c = np.linalg.norm(np.tile([0.25*np.pi, 0, 0, 0], (state_f.shape[0], 1))-state_f, axis=1)
        plt.scatter(x2, y2, s=2, alpha=0.8, c=c)
        plt.colorbar().set_label('Euclidean distance to goal')
        plt.axis('equal')
        plt.title('Final position end effector')

        # up up position
        x, y = pol2cart(1, 0.25*np.pi)
        plt.plot([0, x, x+1], [0, y, y], '-r', linewidth=2)

        return None

    def phase_scatter(self):

        load.init_load()
        data, labels = load.training_set()

        k = 10000
        idx = np.random.randint(0, labels.shape[0], k)

        dof = self.cfg.simulation.dof
        state_i = labels[idx, :2*dof]
        state_f = labels[idx, 2*dof:]
        omega_norm = np.linalg.norm(state_f[:, 2:], axis=1)

        # Plot first link
        plt.subplot(1, 2, 1)
        plt.scatter(state_f[:, 0], state_f[:, 2], s=1, alpha=0.5, c=omega_norm)
        plt.axis("equal")
        plt.xlabel("theta1_f")
        plt.ylabel("omega1_f")

        # Plot second link
        plt.subplot(1, 2, 2)
        plt.scatter(state_f[:, 1], state_f[:, 3], s=1, alpha=0.5, c=omega_norm)
        plt.axis("equal")
        plt.colorbar().set_label("Norm of final velocity")
        plt.xlabel("theta2_f")
        plt.ylabel("omega2_f")

        plt.show()
        return None

    # def costate_angle_scatter(self):
    #     print("Running Coverage")
    #     dof = 1
    #     # Prepare data
    #     load.init_load()
    #     data, labels = load.training_set()
    #     full = np.hstack([data, labels])
    #     full_sorted = full[full[:, 0].argsort()]
    #     data = full_sorted[:, :-(dof*4)]
    #     labels = full_sorted[:, -(dof*4):]

    #     # Random samples to not blow up the computer
    #     idx = np.random.randint(0, data.shape[0], 10000)
    #     d = np.zeros((idx.shape[0], 2))
    #     _, phi = cart2pol(data[idx, 0], data[idx, 1])

    #     for key, val in enumerate(tqdm(idx)):
    #         label_tile = np.tile(labels[val, :], (1000, 1))
    #         costates_hat = self.model.predict(label_tile)
    #         _, phi_hat = cart2pol(costates_hat[:, 0], costates_hat[:, 1])
    #         d[key, :] = _norm.fit(phi_hat)

    #     x = y = np.linspace(-np.pi, np.pi, 100)
    #     plt.plot(x, y, 'b')
    #     plt.scatter(phi, d[:, 0], s=1, alpha=0.3, c=d[:, 1], cmap='autumn')
    #     plt.title('True co-state angle vs. mean of generated co-state angles')
    #     plt.xlabel('phi')
    #     plt.ylabel('phi_hat')
    #     plt.colorbar().set_label('Standard deviation of generated co-state angles')
    #     self.save_fig('costate_angle_scatter')

    #     distance = phi - d[:, 0]
    #     plt.figure()
    #     plt.hist(distance, bins='auto')
    #     fraction = phi[distance < 0.00001].shape[0]/phi.shape[0]
    #     print("Fraction: {} ".format(fraction))
    #     plt.show()
    #     return None

    def save_fig(self, name):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, 'figures', name)
        plt.savefig('{}.pdf'.format(filename))
