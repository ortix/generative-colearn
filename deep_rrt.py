import importlib
import json
import os
import shutil
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from analysis.model import ModelAnalysis
from data.loader import loader as load
from munch import munchify
from planner.node_validator import NodeValidator
from planner.rrt import RRT
from scipy.stats import norm
from settings import settings
from simulation.pendulum import Pendulum
from simulation.robot import Robot
from simulation.robot_2dof import Robot as Robot2Dof
from tqdm import tqdm
from utils.helpers import *  # probably not so smart
from utils.logger import logger

cfg = settings()


class DeepRRT:
    def __init__(self, experiment_n=1):
        # Set warning level and clear terminal
        os.system("cls" if os.name == "nt" else "clear")
        print("|------------ Experiment {} ------------|".format(experiment_n))

        # Local settings
        self.system = cfg.simulation.system
        self.model_name = cfg.model.use
        print("STARTING SIMULATIONS USING MODEL: ", self.model_name)
        self.mode = cfg.simulation.mode
        self.clean = "clean" if cfg.model.clean else "dirty"
        self.run_path = None

        self.load_system(cfg)

        self.planner = RRT(**cfg.planner)

        # Load model and trainer
        self.model = get_model(self.model_name)(**cfg.model[self.model_name].structure)
        self.trainer = get_trainer(self.model_name)(self.model)

        return None

    def load_system(self, cfg):

        if cfg.model.train_only:
            return None

        if self.system == "pendulum":
            self.sim = Pendulum(**cfg.simulation)
        elif self.system == "2dof":
            print("Using old simulation for 2dof")
            self.sim =  Robot2Dof(**cfg.simulation)
        else:
            self.sim = Robot(**cfg.simulation)
        return None

    def load_data(self):
        print("Loading data...")
        try:
            load.init_load(force=True)
        except ValueError as e:
            print(e)
            return False
        return True

    def generate_data(self):
        print("Generating data. Mode: {}".format(self.mode))

        # Run the simulation and get the training data with labels
        self.sim.simulate(cfg.simulation.samples, cfg.simulation.dt)
        self.sim.save(cfg.paths.training)
        if not self.load_data():
            raise ValueError("Can not generate or load data")
        return None

    def plan_path(self):
        # We can pass in a node validator with which we can check whether the
        # node we want to connect to is reachable (model has seen before)
        d_max = cfg.planner.reachability
        if cfg.planner.reachability == -1:
            validator = NodeValidator(
                load.training_labels, d_max, gan_model=self.trainer.get_model()
            )
        else:
            validator = NodeValidator(load.training_labels, d_max)

        # Define initial and final state for n-dof system

        s0 = cfg.planner.state_i
        s1 = cfg.planner.state_f

        self.planner.set_states(s0, s1)
        self.planner.set_predictor(self.trainer.get_model())
        self.planner.set_simulator(self.sim)
        self.planner.set_node_validator(validator)

        n = cfg.planner.runs
        path = self.planner.plan(n, logdir=self.run_path)
        flat_path = np.array(
            [state for trajectory in path[::-1] for state in trajectory]
        )
        np.savetxt(os.path.join(self.run_path, "path_0.txt"), flat_path)
        return None

    def evaluate_model(self):
        """
        Evaluate performance of trained model
        """
        print("Evaluating model")
        model = self.trainer.get_model()
        analyzer = ModelAnalysis(model, self.sim)

        analyzer.set_data(load.test_data, load.test_labels)
        analyzer.set_log_dir(self.run_path)
        analyzer.run()
        return None

    def load_model(self):
        """
        Loads a compiled and pretrained model (model)
        """
        if not cfg.model.load:
            return False
        model_weights = os.path.join(
            cfg.paths.models,
            "{}_{}_{}_{}_weights.h5".format(
                self.model_name, self.system, self.mode, self.clean
            ),
        )

        if not self.model_name:
            print("No models found to load...")
            return False
        try:
            print("Loading weights {}".format(model_weights))
            self.trainer.set_model_weights(model_weights)
        except Exception as e:
            print(repr(e))
            return False
        return True

    def train_model(self):
        self.trainer.train(**cfg.model[self.model_name].training)

        # @Todo: Save model somewhere else, do not depend on specific API here!
        if cfg.model.save:
            decoder = self.trainer.get_model()
            model_file = os.path.join(cfg.paths.models, self.model_name)
            decoder.save_weights(
                "{}_{}_{}_{}_weights.h5".format(
                    model_file, self.system, self.mode, self.clean
                )
            )
        return None

    def write_logs(self):
        filename = os.path.join(self.run_path, "log.json")
        with open(filename, "w") as outfile:
            json.dump(logger, outfile, indent=4)

        # We copy the settings file to the run path as well for reference
        # cwd = os.path.dirname(os.path.realpath(__file__))
        # settings = os.path.join(cwd, 'settings.json')
        # settings_new = os.path.join(self.run_path, 'settings.json')
        # print("Copying {} to {}".format(settings, settings_new))
        # shutil.copy2(settings, settings_new)
        return None

    def clean_up_irq(self):
        """
        Clean up if interrupted
        """
        print("Removing dir: {}".format(self.run_path))
        shutil.rmtree(self.run_path)
        exit()

    def clean_up(self):
        """
        General cleanup routine
        """
        # Clear logger after each run
        logger.clear()

        try:
            os.rmdir(self.run_path)
            print("Directory {} was empty. Removed.".format(self.run_path))
        except OSError:
            pass

        # Close tensorflow session if possible
        try:
            self.model.close_session()
            print("Tensorflow Session closed")
        except:
            # Only close when tensorflow model is used
            pass
        return None

    def move_run_to(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.move(self.run_path, path)
        return os.path.join(path, os.path.basename(os.path.normpath(self.run_path)))

    def run(self, experiment_n=1):
        try:
            self.run_path = create_run_dir(cfg.paths.tmp, self.model_name, self.clean)
            if not cfg.simulation.load or not self.load_data():
                self.generate_data()

            if not cfg.model.load or not self.load_model():
                self.train_model()

            if cfg.model.train_only:
                return None
            self.evaluate_model()
            self.plan_path()
            self.write_logs()
            self.clean_up()
        except KeyboardInterrupt:
            print("Interrupted. Starting clean-up...")
            self.clean_up_irq()
        return None
