import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm as _norm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from settings import settings as cfg
from sklearn.metrics import mean_squared_error as mse
from utils.logger import logger


class ModelAnalysis:
    def __init__(self, model, sim):
        self.model = model
        self.sim = sim
        self.cfg = cfg()

        self.data = []
        self.labels = []
        self.log_dir = self.cfg.paths.tmp
        self.samples = []
        return None

    def set_log_dir(self, path):
        self.log_dir = path

    def set_data(self, data, labels):
        self.data = data
        self.labels = labels
        return None

    def generate_samples(self):
        print("Generating samples...")
        self.samples = self.model.predict(self.labels)
        return None

    def save_fig(self, name):
        model = self.cfg.model.use
        svg = os.path.join(self.log_dir, "{}_{}.pdf".format(model, name))
        png = os.path.join(self.log_dir, "{}_{}.png".format(model, name))
        plt.savefig(svg, bbox_inches="tight")
        plt.savefig(png, bbox_inches="tight")

    def plot_costate_concentration(self):
        """
        For each n trajectories generate m costates and plot them with the corresponding
        costates. The generated costates should be densely populated around real costate 
        """
        if self.sim.dof > 1:
            print("Not plotting costate concentration")
            return None
        n = 3
        m = 500
        idx = np.random.randint(0, self.labels.shape[0], n)
        costates = self.data[idx, :]
        labels = self.labels[idx, :]
        costates_hat = np.zeros((m, self.data.shape[1]))

        plt.figure()
        c = plt.Circle((0, 0), radius=1, fill=False, alpha=0.5)
        # plt.scatter(self.data[:, 0], self.data[:, 1], c='#dddddd', s=0.1)
        plt.axis("equal")
        plt.gca().add_artist(c)
        # plt.title('True vs Generated costate concentration')
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\mu$")
        colors = iter({"r", "g", "b"})

        for i in range(n):
            label_rep = np.tile(labels[i, :], (m, 1))
            costates_hat = self.model.predict(label_rep)
            color = next(colors)
            plt.scatter(costates[i, 0], costates[i, 1], marker="*", color=color, s=80)
            plt.scatter(
                costates_hat[:, 0], costates_hat[:, 1], s=1, color=color, alpha=0.2
            )
        self.save_fig("costate_concentration")

        return None

    def plot_cost_qq(self):
        """
        We can scatter the variables against each other because the generated values
        are generated with labels that are already linked with the true values
        """
        x = y = np.linspace(0, 1.5, 100)
        idx = np.random.randint(0, self.data.shape[0], 1000)
        plt.figure()
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        plt.suptitle("Cost and time discrepancy plot", y=0.95)
        ax[0].scatter(self.samples[idx, -1], self.data[idx, -1], s=0.5)
        ax[0].plot(x, y, "r")
        ax[0].set_xlabel(r"$\hat{t_f} \quad (s)$")
        ax[0].set_ylabel(r"$t_f \quad (s)$")
        ax[1].scatter(self.samples[idx, -2], self.data[idx, -2], s=0.5)
        ax[1].plot(x, y, "r")
        ax[1].set_xlabel(r"$\hat{c}$")
        ax[1].set_ylabel(r"$c$")
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        self.save_fig("cost_qq")

        # Print cost and t_f MSE:
        logger["model"]["t_f_mse"] = mse(self.samples[idx, -1], self.data[idx, -1])
        logger["model"]["cost_mse"] = mse(self.samples[idx, -2], self.data[idx, -2])
        print(
            "[Cost MSE]: {} \t [t_f MSE]: {}".format(
                logger["model"]["cost_mse"], logger["model"]["t_f_mse"]
            )
        )

        return None

    def plot_2d_costate_scatter(self):
        idx = np.random.randint(0, self.data.shape[0], 2500)
        costates_hat = self.samples[idx, :]
        plt.figure()
        # Draw circle and scatter the costates over it
        c = plt.Circle((0, 0), radius=1, fill=False, alpha=0.5)
        plt.scatter(costates_hat[:, 0], costates_hat[:, 1], s=0.5, alpha=0.95)
        plt.gca().add_artist(c)

        plt.axis("equal")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\mu$")
        mode = self.cfg.simulation.mode
        plt.title("Proximity of generated costates to unit circle")
        self.save_fig("costate_unit_circle")

    def plot_costate_proximity(self):
        if self.cfg.simulation.dof == 1:
            self.plot_2d_costate_scatter()

        norm = np.linalg.norm(self.samples[:, :-2], axis=1)
        n = int(self.labels.shape[1] / 2)
        plt.figure()
        plt.hist(norm, bins="auto")
        plt.title("Proximity of costates to {}-D hyperpshere ".format(n))
        plt.xlabel("Norm of points")
        # filename = os.path.join(self.log_dir, "costate_proximity_histogram.png")
        self.save_fig("costate_proximity_histogram")
        mean, std = _norm.fit(norm)
        print("[Proximity] mu: {} \t std: {}".format(mean, std))
        logger["model"]["proximity"] = float(mean)
        logger["model"]["proximity_std"] = float(std)
        return None

    def plot_costate_angles(self):
        if int(self.cfg.simulation.dof) != 1:
            return None
        costates_hat = self.samples
        costates = self.data
        angles_hat = np.arctan2(-costates_hat[:, 0], -costates_hat[:, 1])
        angles = np.arctan2(-costates[:, 0], -costates[:, 1])
        mode = self.cfg.simulation.mode
        plt.figure()
        plt.title("Distribution of costate angles for {} based model".format(mode))
        plt.hist(angles, bins=100)
        plt.hist(angles_hat, bins=100, alpha=0.7)
        plt.legend(["True", "Generated"])
        filename = os.path.join(self.log_dir, "angle_distribution.png")
        plt.savefig(filename)
        return None

    def plot_generated_distribution(self):
        costates_hat = self.samples
        costates = self.data
        n = self.samples.shape[1]
        dof = self.cfg.simulation.dof
        fig, axes = plt.subplots(2, int(n / 2), figsize=(10, 8))
        mode = self.cfg.simulation.mode
        plt.suptitle("Distribution for {} based model".format(mode))

        for ax, i in zip(axes.flat[:], range(n)):
            costate = (
                "lambda {}".format(i + 1) if i < dof else "mu {}".format(i - dof + 1)
            )
            if i == n - 2:
                costate = "cost"
            if i == n - 1:
                costate = "time"
            mean, std = _norm.fit(costates_hat[:, i])
            ax.hist(costates[:, i], bins="auto")
            ax.hist(costates_hat[:, i], bins="auto", alpha=0.7)
            ax.legend([costate, "{}_hat".format(costate)])
            ax.set_xlabel(costate)
            ax.set_title(
                r"$\mu={:4.3f} \quad \sigma^2={:4.3f}$".format(mean, std ** 2),
                fontsize=8,
            )
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        self.save_fig("data_distribution")
        return None

    def torque_switch_factor(self):
        from data.loader import loader as load

        print("Calculating torque switching factor")

        data_set, labels_set = load.training_set()
        n = 10000  # labels_set.shape[0]
        idx = np.random.randint(0, labels_set.shape[0], n)
        labels = labels_set[idx, :]

        states = np.split(labels, 2, axis=1)
        costates = data_set[idx, :]

        switches = 0

        if self.sim.dof == 2:
            for i in tqdm(range(n)):
                mu0 = costates[i, -2:]
                _, _, _, mu1, _ = self.sim.simulate_steer_full(
                    states[0][i, :], costates[i, :]
                )
                if not np.any(np.sign(mu0) + np.sign(mu1)):
                    switches += 1

        # This is done in a stupid way. I could've just checked whether the sign is different
        # between mu0 and mu1...
        if self.sim.dof == 1:
            for i in tqdm(range(n)):
                _, _, s = self.sim.simulate_steer_full(states[0][i, :], costates[i, :])
                if int(np.abs(np.sign(s[:, 3]).sum())) != s[:, 3].shape[0]:
                    switches += 1

        print("Switches: {} \t Data points: {} ".format(switches, n))
        logger["model"]["switch_factor"] = switches / n
        return None

    def error_rate(self):
        print("Computing model error rate. This might take 10s to start...")
        n = 1000
        m = 1

        idx = np.random.randint(0, self.labels.shape[0], n)

        # Repeat the labels to generate multiple costates for single trajectory
        labels = np.repeat(self.labels[idx, :], m, axis=0)
        costates_hat = self.model.predict(labels)

        # First half of labels contains the initial states, second half are final states
        states = np.split(labels, 2, axis=1)
        costates = self.data[idx, :]

        theta_hat = []
        omega_hat = []
        dof = self.cfg.simulation.dof
        for i in tqdm(range(m * n)):
            state, _ = self.sim.simulate_steer(states[0][i, :], costates_hat[i, :])
            theta_hat.append(state[:dof])
            omega_hat.append(state[dof:])

        theta, omega = np.split(
            states[1], 2, axis=1
        )  # pylint: disable=unbalanced-tuple-unpacking
        theta_omega_hat = np.column_stack((theta_hat, omega_hat))

        theta_e = mse(theta, theta_hat)
        omega_e = mse(omega, omega_hat)
        states_e = mse(states[1], theta_omega_hat)

        p_std = np.std(np.abs(theta - theta_hat).flatten())
        v_std = np.std(np.abs(omega - omega_hat).flatten())

        print("Total MSE: {}".format(states_e))
        print("Position MSE: {} \t \t Velocity MSE: {}".format(theta_e, omega_e))
        print("Position std: {} \t \t Velocity std: {}".format(p_std, v_std))
        logger["model"]["mse_length"] = n
        logger["model"]["total_mse"] = states_e
        logger["model"]["position_mse"] = theta_e
        logger["model"]["velocity_mse"] = omega_e
        logger["model"]["position_std"] = p_std
        logger["model"]["velocity_std"] = v_std

    def valid_alpha_rate(self):
        print("Computing valid alpha rate")
        n = self.labels.shape[0]
        costates_hat = self.samples
        valid = 0
        for idx in tqdm(range(n)):
            states = self.labels[idx, :2]
            costates = costates_hat[idx, :2]
            if self.sim.validate_costates(states, costates):
                valid += 1
        valid_rate = valid / n * 100
        print(
            "Alpha accuracy: {}% \t Invalid amount: {}/{}".format(
                valid_rate, n - valid, n
            )
        )

    def plot_trajectories(self):
        if not type(self.model).__name__ == "KNN":
            return self.plot_trajectories_gan()
        return self.plot_trajectories_knn()

    def plot_trajectories_gan(self):
        from data.loader import loader as load
        from matplotlib import cm

        data = load.training_data
        labels = load.training_labels

        # Load true trajectory
        i = np.random.randint(0, self.labels.shape[0])
        true_tr = self.labels[i, :]
        true_cs = self.data[i, :]

        # Predict costates and simulate trajectory
        pred_cs = self.model.predict([true_tr])

    def plot_trajectories_knn(self):
        from data.loader import loader as load
        from matplotlib import cm

        data = load.training_data
        labels = load.training_labels

        # Load true trajectory
        i = np.random.randint(0, self.labels.shape[0])
        true_tr = self.labels[i, :]
        true_cs = self.data[i, :]

        # Predict costates and simulate trajectory
        pred_cs = self.model.predict([true_tr])[0]
        pred_tr, _ = self.sim.simulate_steer(true_tr[:2], pred_cs[:3])

        # Plot simulation and prediction
        plt.figure()
        plt.plot([true_tr[0], true_tr[2]], [true_tr[1], true_tr[3]])
        plt.plot([true_tr[0], pred_tr[0]], [true_tr[1], pred_tr[1]])
        plt.scatter(true_tr[0], true_tr[1], marker="o", c="C0")
        plt.scatter(true_tr[2], true_tr[3], marker="d", c="C0")
        plt.scatter(pred_tr[0], pred_tr[1], marker="x", c="C1")

        # Get nearest neighbors to the true trajectory
        dist, nn_idx = self.model.model.kneighbors([true_tr], n_neighbors=10)
        dist_norm = np.copy(dist[0])
        dist_norm *= 1.0 / dist_norm.max()
        nn_cs = data[nn_idx[0]]
        for i, cs in enumerate(nn_cs):
            pred_tr, _ = self.sim.simulate_steer(true_tr[:2], cs[:3])
            # plt.plot([true_tr[0], pred_tr[0]], [true_tr[1], pred_tr[1]], c=plt.cm.cool(dist_norm[i]), alpha=0.7)

        plt.grid(True)
        filename = os.path.join(self.log_dir, "trajectories.png")
        plt.savefig(filename)

    def plot_knn_spread(self):
        if not type(self.model).__name__ == "KNN":
            print("Not running plot_knn_spread")
            return
        from data.loader import loader as load
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Load the training data since that is what the model has
        data = load.training_data

        labels = load.training_labels
        neighbors = 15
        i = np.random.randint(0, self.labels.shape[0])
        costates_hat = self.model.predict([self.labels[i, :]])
        dist, idx = self.model.model.kneighbors([self.labels[i, :]], neighbors)
        costates_nn = data[idx[0], :]
        plt.figure()
        c = plt.Circle((0, 0), radius=1, fill=False, alpha=0.5)
        plt.axis("equal")
        plt.gca().add_artist(c)
        plt.subplot(111)
        plt.scatter(
            costates_nn[:, 0],
            costates_nn[:, 1],
            s=100,
            c=dist[0],
            cmap="autumn",
            alpha=0.8,
        )
        plt.colorbar().set_label("Euclidean distance to query")
        plt.scatter(self.data[i, 0], self.data[i, 1], s=50, c="k", marker="*")
        plt.scatter(costates_hat[:, 0], costates_hat[:, 1], s=20, marker="x", c="k")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\mu$")
        plt.ylim(-1.2, 1.2)
        plt.xlim(-1.2, 1.2)
        plt.title("{} neighboring costates spread over unit circle".format(neighbors))
        plt.legend(["Neighboring costates", "True costate", "Predicted costate"])
        self.save_fig("knn_test")
        return None

    def costate_error(self):
        n = 1000
        n_dof = int(self.sim.dof)
        idx = np.random.randint(0, self.labels.shape[0], n)
        # states = np.split(self.labels[idx, :], 2, axis=1)
        costates = np.split(self.data[idx, : (n_dof * 2)], 2, axis=1)
        costates_hat = np.split(self.samples[idx, : (n_dof * 2)], 2, axis=1)

    def run(self):
        self.generate_samples()
        self.plot_costate_concentration()
        self.plot_knn_spread()
        self.plot_costate_proximity()
        self.plot_generated_distribution()
        self.plot_cost_qq()
        # self.plot_trajectories()
        # self.plot_costate_polar()
        # self.valid_alpha_rate()
        # self.plot_costate_angles()
        # Run last, slow
        # self.costate_error()
        # self.torque_switch_factor()
        # self.error_rate()
        plt.close("all")
        return None


if __name__ == "__main__":
    pass
