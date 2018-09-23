# Imports
import copy
import os
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from planner.node import Node
from utils.logger import logger
from settings import settings
cfg = settings()


class MaxInvalidAlphaError(Exception):
    pass


class RRT():
    def __init__(self, **kwargs):
        '''
        goal_bias: percentage of bias to select goal
        max_nodes: terminate rrt after max nodes is reached
        sampling_bounds: array with n arrays containing min max for sampling
        threshold: Euclidean distance between random and goal state
        '''
        self.n_states = 0
        self.start = []
        self.goal = []
        self.node_list = []
        self.goal_selected = None

        self._predictor = None
        self._sim = None
        self._validator = None

        self.max_nodes = cfg.planner.max_nodes
        self.goal_bias = cfg.planner.goal_bias
        self.threshold = cfg.planner.threshold

        # Logging
        self.prediction_time = []
        self.log = {
            "attempts": 0,
            "total_time": 0.0,
            "planning_time": [],
            "n_nodes": [],
            "prediction_time": {"mu": [], "median": [], "std": []},
            "accuracy": {"mu": [], "median": [], "std": []},
            "fail_rate": 0.0
        }
        return None

    def reset(self):
        self.node_list = []
        plt.close('all')
        return None

    def set_states(self, start, goal):
        self.start = Node(start)
        self.goal = Node(goal)
        print("RRT Trajectory: {} --> {}".format(start, goal))
        return None

    def set_predictor(self, predictor):
        self._predictor = predictor

    def set_simulator(self, simulator):
        self._sim = simulator
        self.n_states = self._sim.dof*2

    def set_node_validator(self, validator):
        self._validator = validator

    def get_steering(self, current, random):
        '''
        Predict the steering action from current to random and the distance
        Return costates, cost (distance) and t_f
        '''
        states = np.hstack([current, random])
        # Do not time during experimental run
        start = timer()
        prediction = self._predictor.predict(states)
        self.prediction_time.append(timer()-start)

        # We need to add randomness to our prediction to be random complete
        # Clip the generated cost-to-go
        euc = np.linalg.norm(current-random, axis=1)
        prediction[:, -2] = prediction[:, -2].clip([10e-5*euc[:]], 10e5*euc[:])
        prediction[:, -1] = prediction[:, -1].clip([10e-5*euc[:]], 10e5*euc[:])

        # Add Gaussian noise to the costates
        if self.goal_selected:
            prediction[:, :-2] = np.random.normal(prediction[:, :-2], scale=cfg.planner.goal_noise_var)

        return prediction

    def sample_random_costate(self):
        n = self._sim.dof*2
        point = np.array([np.random.normal() for i in range(n)])
        point = point/np.linalg.norm(point)
        return point

    def sample_random_node(self):
        self.goal_selected = False
        if np.random.randint(0, 100) < self.goal_bias:
            self.goal_selected = True
            return self.goal.state
        state_random = np.zeros((self.n_states, ))
        for n in range(self.n_states):
            state_random[n] = np.random.uniform(
                self._sim.sampling_bounds[n][0],
                self._sim.sampling_bounds[n][1])
        return state_random

    def filter_valid_nodes(self, state_random):
        states = np.array([node.state for node in self.node_list])
        idx = self._validator.validate(states, state_random)
        return idx

    def steer(self, valid_idx, state_random):
        states = np.array([self.node_list[i].state for i in valid_idx])
        random_tile = np.tile(state_random, (states.shape[0], 1))
        u = self.get_steering(states, random_tile)

        # Select node for expansion by finding the trajectory with the lowest cost.
        # We take the 3 cheapest nodes and select 1 random
        # This is effectively disabled so minimum cost node is selected
        if u[:, -2].shape[0] > 1e3:
            k = 3
            local_idx = np.random.choice(np.argpartition(u[:, -2], k)[:k])
        else:
            local_idx = np.argmin(u[:, -2])  # select absolute minimum
        u_min = u[local_idx, :]
        idx = valid_idx[local_idx]
        current = self.node_list[idx].state

        final_state, trajectory = self._sim.simulate_steer(current, u_min)

        return final_state, idx, trajectory

    def plan_single(self):
        fail = False
        delta = []
        self.prediction_time = []
        self.node_list.append(self.start)
        start = timer()
        omega_min = 9999
        theta_min = 9999
        pbar = tqdm(total=self.max_nodes)
        while True:
            state_random = self.sample_random_node()

            valid_idx = self.filter_valid_nodes(state_random)
            if len(valid_idx) == 0:
                continue

            new_node_state, parent_idx, trajectory = self.steer(valid_idx, state_random)

            delta.append(((state_random - new_node_state)**2).mean(axis=0))

            # Connect nearest neighbor to new node
            new_node = Node(new_node_state)
            new_node.parent = parent_idx
            new_node.trajectory = trajectory

            # Append nodeList
            self.node_list.append(new_node)

            # Check if goal is reached with margin
            dof = self._sim.dof
            d = np.linalg.norm(new_node.state - self.goal.state)
            theta = np.linalg.norm(new_node.state[:dof] - self.goal.state[:dof])
            omega = np.linalg.norm(new_node.state[dof:] - self.goal.state[dof:])
            if theta < theta_min:
                theta_min = theta
                omega_min = omega

            # Information
            if not cfg.planner.debug:
                pbar.update(1)
            if len(self.node_list) % 10 == 0 and cfg.planner.debug:
                print("theta_min: {:.3f} \t omega_min {:.3f} \t Delta: {:.3f} \t d: {} \t Nodes: {} \t Goal: {}".format(
                    theta_min,
                    omega_min,
                    delta[-1],
                    d,
                    len(self.node_list),
                    "Selected" if self.goal_selected else "Not Selected"
                ))
            # self.visualize(state_random)
            if cfg.planner.debug:
                self.draw_all(state_random, new_node, draw_prediction=True)

            if len(self.node_list) > self.max_nodes:
                pbar.close()
                print("Max iterations reached. Exiting")
                fail = True
                break

            if d <= self.threshold and d >= 0.0:
                pbar.close()
                self.log_delta(delta)
                self.log_prediction_time(self.prediction_time)
                print("Current state: {} \t Goal: {} ".format(new_node.state, self.goal.state))
                print("Goal!")
                break

        # path = [self.goal.state]
        path = []
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append(node.trajectory)
            last_index = node.parent
        path.append(np.array([self.start.state]))
        return path, timer()-start, fail

    def visualize(self, state_random):
        import requests
        import json
        r = requests.get('http://127.0.0.1:7777/get_joint_angles')
        model = json.loads(r.text)
        model["angles"][0] = state_random[0]
        model["angles"][1] = state_random[1]
        payload = json.dumps(model, separators=(',', ':'))
        r = requests.post('http://127.0.0.1:7777/set_joint_angles', payload,
                          headers={'Content-type': 'application/json'})
        return None

    def log_prediction_time(self, time):
        self.log["prediction_time"]["mu"].append(np.mean(time))
        self.log["prediction_time"]["median"].append(np.median(time))
        self.log["prediction_time"]["std"].append(np.std(time))
        return None

    def log_delta(self, delta):
        self.log["accuracy"]["mu"].append(np.mean(delta))
        self.log["accuracy"]["median"].append(np.median(delta))
        self.log["accuracy"]["std"].append(np.std(delta))
        return None

    def plan(self, n=1, logdir="./"):
        run = 0
        start = timer()
        while run < n:
            self.log["attempts"] += 1
            if self.log["attempts"] > n*2:
                print("Fail rate 50\% ")
                break
            path, time, fail = self.plan_single()
            if fail:
                self.reset()
                continue
            self.log["planning_time"].append(time)
            self.log["n_nodes"].append(len(self.node_list))
            filename = os.path.join(logdir, "rrt_{}.pdf".format(run))
            if cfg.planner.plotting:
                self.draw_all()
                self.draw_path(path, filename)
            self.print_log()
            self.reset()
            run += 1
        self.log["fail_rate"] = (1-(n/self.log["attempts"]))*100
        self.log["total_time"] = timer()-start
        print("[Fail rate]: {}".format(self.log["fail_rate"]))
        print("[Total Time]: {}".format(self.log["total_time"]))
        logger["rrt"] = self.log
        return path

    def draw_all(self, rnd=None, nearest=None, draw_prediction=False):
        import matplotlib.colors as cl
        import matplotlib.cm as cm
        
        def inside_bound(state):
            return True

        # Force colorbar to show the velocity as the node colors
        omegas = np.array([node.state for node in self.node_list  if inside_bound(node.state)])[:, self._sim.dof:]
        omegas_norm = np.linalg.norm(omegas, axis=1)
        vmax = omegas_norm.max()
        omegas_normal = cl.Normalize(vmax=vmax)(omegas_norm)
        cmap = cm.get_cmap('winter')
        colors = cmap(omegas_normal)
        Z = [[0, 0], [0, 0]]
        levels = np.linspace(0, vmax, 100)
        if self._sim.dof > 1:
            colorbar_ax = plt.contourf(Z, levels, cmap=cmap)
        plt.clf()

        # Which states to plot
        idx = [0, 1]  # [1, 3] for lower link in 2dof

        if rnd is not None:
            plt.plot(rnd[idx[0]], rnd[idx[1]], "^y")
        for c, node in zip(colors, self.node_list):
            if node.parent is not None:
                plt.plot(node.trajectory[:, idx[0]], node.trajectory[:, idx[1]], "-k", linewidth=0.8, alpha=0.5)
                if self._sim.dof == 1:
                    plt.plot(node.state[idx[0]], node.state[idx[1]], "ok", markersize=3, markerfacecolor=(0, 0, 0, 0.5))
                else:
                    plt.plot(node.state[idx[0]], node.state[idx[1]], "o", markersize=3, color=c)

        # Only useful when plotting during RRT, this shows the error of expansion
        if cfg.planner.debug:
            expansion = self.node_list[-1]
            plt.plot([expansion.state[idx[0]], self.node_list[expansion.parent].state[idx[0]]], [
                expansion.state[idx[1]], self.node_list[expansion.parent].state[idx[1]]], "-b", linewidth=1.0)

        if nearest is not None:
            plt.plot(nearest.state[idx[0]], nearest.state[idx[1]], "or")

        plt.plot(self.start.state[idx[0]], self.start.state[idx[1]], "or")
        plt.plot(self.goal.state[idx[0]], self.goal.state[idx[1]], "og")
        if self._sim.dof > 1:
            plt.colorbar(colorbar_ax).set_label("Norm of omega")
        plt.axis([
            1.2*self._sim.sampling_bounds[idx[0]][0],
            1.2*self._sim.sampling_bounds[idx[0]][1],
            1.2*self._sim.sampling_bounds[idx[1]][0],
            1.2*self._sim.sampling_bounds[idx[1]][1]])
        plt.grid(True)
        if draw_prediction:
            plt.pause(0.001)

    def draw_path(self, path_full, filename):
        path = np.array(path_full)
        run = len(self.log["n_nodes"])-1
        time = self.log["planning_time"][run]
        n_nodes = self.log["n_nodes"][run]
        # title = "Run {} in {} sec with {} nodes".format(run, round(time, 4), n_nodes)
        for trajectory in path:
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-r', linewidth=1)
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'or', markersize=3)

        plt.grid(True)
        # plt.title(title)
        if self._sim.dof == 1:
            plt.xlabel(r'$\theta$ (in rad)')
            plt.ylabel(r'$\omega$ (in rad s$^{-1}$)')
        else:
            plt.xlabel(r'$\theta_1$')
            plt.ylabel(r'$\theta_2$')

        plt.savefig(filename, bbox_inches='tight')

    def print_log(self):
        print("Completed {}/{} runs successfully...".format(
            len(self.log["planning_time"]),
            self.log["attempts"]
        ))
        print("---------------")
        print("[Reachability] {}".format(self._validator.reachability_log))
        print("[Model] {}".format(self._predictor.__class__.__name__))
        print("[Time] mu: {:.3f} \t median {:.3f} \t std: {:.3f}".format(
            np.mean(self.log["planning_time"], axis=0),
            np.median(self.log["planning_time"], axis=0),
            np.std(self.log["planning_time"], axis=0)))
        print("[Nodes] mu: {:.1f} \t median: {:.1f} \t std: {:.1f}".format(
            np.mean(self.log["n_nodes"], axis=0),
            np.median(self.log["n_nodes"], axis=0),
            np.std(self.log["n_nodes"], axis=0)))
        print("[Accuracy] mu: {:.3f} \t median: {:.3f} \t std: {:.3f}".format(
            np.mean(self.log["accuracy"]["mu"], axis=0),
            np.median(self.log["accuracy"]["median"], axis=0),
            np.linalg.norm(self.log["accuracy"]["std"], axis=0)))
        print("---------------")
        print("\n")
        return None


class Simulator():

    def __init__(self):

        return None

    def simulate_steer(self, s0, u, dt=0.01):
        # Define vector between random node and nearest neighbor
        # u[:2] = random state
        # u[-1] = cost
        # s0 = nearest node state
        support_vector = (u[:2]-s0)/np.linalg.norm(u[:2]-s0)

        # Create node on the line pointing from nn to random node
        expansion_vector = 0.3*support_vector
        s1 = s0+expansion_vector
        return s1


class Predictor():
    def __init__(self):
        self.latent_size = 0
        return None

    def predict(self, states):
        '''
        Return the costate and distance between current and random state.
        In this dummy implementation the costate vector is just the random state with noise
        '''
        # Shitty dummy data generation
        costates = np.random.normal(0, 0.01, (1, 2)) + states[2:]
        cost = np.linalg.norm(states[:2] - states[2:])
        cost = np.expand_dims([cost], axis=0)
        prediction = np.hstack([costates, cost])
        return prediction


if __name__ == "__main__":

    planner = RRT()
    planner.set_states([0, 0], [2, 2])
    planner.set_predictor(Predictor())
    planner.set_simulator(Simulator())
    path = planner.plan(1)

    # plt.show()
