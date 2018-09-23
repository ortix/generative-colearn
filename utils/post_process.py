import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from settings import settings
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def nested_dict(): return defaultdict(nested_dict)


class Pipeline():

    def __init__(self):
        self.cfg = settings()
        self.logs = []
        self.exp_tabs = 10
        self.missing_keys = []
        return None

    def reset(self):
        self.logs = []

    def consolidate(self, model, mode_n=0, model_offset=4, subfolder=['results'], clean=None):
        # Create a list of pattern matched filenames to load
        if not clean:
            prefix = "{}_*/log.json".format(model)
        else:
            prefix = "{}_{}_*/log.json".format(model, clean)

        if(model == "clsgan"):
            mode_n += model_offset

        # Some experiments are stored within a mode subdir e.g. during a sweep
        mode = 'mode_{}'.format(mode_n) if mode_n >= 0 else '.'
        pattern = os.path.join(self.cfg.paths.tmp,
                               *subfolder,
                               mode, prefix)

        log_files = glob.glob(pattern)

        if not len(log_files):
            print("Could not find any log files with pattern {}".format(pattern))
            return False

        for log_file in log_files:
            # print("Opening: {}".format(log_file))
            with open(log_file, 'r') as f:
                self.logs.append(json.load(f))
        print("Loaded {} log files".format(len(self.logs)))

        return True

    def compute_rrt_stats(self):
        print("\n[Computing RRT Statistics]")
        # Single var data
        data = self.get_single_var_data("rrt")
        for key in data.keys():
            self.print_var(data[key], key)

        # Multi var list
        data = self.get_list_data("rrt")
        for key in data.keys():
            self.print_var(data[key], key)

        # Multi var dict
        data = self.get_dict_data("rrt")
        for key in data.keys():
            self.print_stats(data[key], key)

        return None

    def get_single_var_data(self, stat_type):
        """
        Returns the log variables that only have 1 entry per run such as fail rate or proximity
        """
        keys = [k for (k, v) in self.logs[0][stat_type].items() if not isinstance(v, (dict, list))]
        data = nested_dict()
        for key in keys:
            data[key] = [log[stat_type][key] for log in self.logs]
        return data

    def get_dict_data(self, stat_type):
        """
        Returns the log variables that are stored with dicts containing mu, std, and med
        """
        keys = [k for (k, v) in self.logs[0][stat_type].items() if isinstance(v, (dict))]
        data = nested_dict()
        for key in keys:
            for stat in ["mu", "std", "median"]:
                data[key][stat] = [item for log in self.logs for item in log[stat_type][key][stat]]
        return data

    def get_list_data(self, stat_type):
        """
        Returns the log variables that are stored as lists e.g. n_nodes
        Un-flattened is returned (list of list per epoch) as second output
        """
        keys = [k for (k, v) in self.logs[0][stat_type].items() if isinstance(v, (list))]
        data = nested_dict()
        for key in keys:
            data[key] = [item for log in self.logs for item in log[stat_type][key]]
        return data

    def compute_model_stats(self):
        try:
            data = self.get_single_var_data("model")
        except KeyError:
            print("Missing model stats")
            return None
        
        print("\n[Computing Model Statistics]")
        keys = ["cost_mse", "t_f_mse", "total_mse", "position_mse", "velocity_mse"]
        for key in keys:
            print("[{}]\t {:.3f}".format(key, self.total_mse(
                data[key], data["mse_length"])).expandtabs(self.exp_tabs))

        # Print the statistic for the model
        for key in ["proximity", "reachability"]:
            print("[{}]\t mu: {:.3f} \t std: {:.3f}".format(key,
                                                            np.mean(data[key]),
                                                            np.linalg.norm(data["{}_std".format(key)])
                                                            ).expandtabs(self.exp_tabs))

        return None

    def total_mse(self, mse, mse_length=1000):
        mse_length = np.median(mse_length)
        return np.sqrt(np.sum(np.square(mse)*mse_length)/(len(mse)*mse_length))

    def print_mode(self):
        try:
            print(self.logs[0]["experimental_mode"])
        except:
            pass

    def print_stats(self, data, name):
        """
        Print statistics of a dict of data already containing mu, std, median
        """
        print("[{}]\t mu: {:.4f} \t std: {:.4f} \t med: {:.4f}".format(
            name,
            np.mean(data["mu"]),
            np.linalg.norm(data["std"]),
            np.median(data["median"])
        ).expandtabs(self.exp_tabs))

    def print_var(self, data, name):
        """
        Print statistics of a list of data
        """
        print("[{}]: \t mu: {:<4.4f} \t std: {:<4.4f} \t med: {:<4.4f}".format(
            name,
            np.mean(data),
            np.std(data),
            np.median(data),
        ).expandtabs(self.exp_tabs))

    def reachability_box_plot(self):
        """
        Plot the effect of reachability on accuracy
        """
        # fig = plt.figure(figsize=(9, 3))
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
        acc = {"knn": [], "clsgan": []}
        nodes = {"knn": [], "clsgan": []}
        fail = {"knn": [], "clsgan": []}
        labels = []
        for key, model in enumerate(["clsgan", "knn"]):
            for n in range(10):
                model_mod = "knn" if model == "knn" else model
                self.consolidate(model,
                                 mode_n=n,
                                 model_offset=0,
                                 subfolder=["experimental", "reach_sweep_2", "reach_sweep_{}".format(model_mod)])

                if key == 0:
                    labels.append(self.logs[0]["experimental_mode"]["reachability"])
                acc[model].append(self.get_dict_data("rrt")["accuracy"]["median"])
                nodes[model].append(self.get_list_data("rrt")["n_nodes"])
                fail[model].append(self.get_single_var_data("rrt")["fail_rate"])
                self.logs = []

        def draw_plot(axes, data, offset, edge_color, fill_color):
            pos = np.arange(len(data))+offset
            axes.yaxis.grid(True, which='major', alpha=0.6)
            axes.set_yscale("log")
            bp = axes.boxplot(data, positions=pos, sym='+', widths=0.3, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)
            return bp
        bp = []
        bp.append(draw_plot(ax[0], nodes["clsgan"], -0.2, "black", "steelblue"))
        bp.append(draw_plot(ax[0], nodes["knn"], +0.2, "black", "mistyrose"))
        bp.append(draw_plot(ax[1], acc["clsgan"], -0.2, "black", "steelblue"))
        bp.append(draw_plot(ax[1], acc["knn"], +0.2, "black", "mistyrose"))

        labels = [np.round(label[0], 2) for label in labels]

        for i, (fail_knn, fail_gan) in enumerate(zip(fail["knn"], fail["clsgan"])):
            ax[0].text(i+(-0.2), 1200, "{}%".format(int(fail_gan[0])), size="small", horizontalalignment='center')
            ax[0].text(i+(+0.23), 1200, "{}%".format(int(fail_knn[0])), size="small", horizontalalignment='center')

        ax[0].set_ylabel("Nodes")
        ax[0].set_ylim(10, 2000)
        ax[1].set_xticks(np.arange(len(labels)))
        ax[1].set_xticklabels(labels)
        ax[1].set_xlabel("Reachability")
        ax[1].set_ylabel("Steering error")
        ax[1].legend([bp[-2]["boxes"][0], bp[-1]["boxes"][0]], ["GAN", "KNN"])
        # plt.show()
        self.save_fig("reachability_box")
        return None

    def save_fig(self, name):
        filename = os.path.join(self.cfg.paths.figs, "{}.pdf".format(name))
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(filename, bbox_inches="tight")
        print("Figure saved in: {}".format(filename))

    def clean_sweep_stats(self):
        """
        Plot the effect of reachability on accuracy
        """
        acc = []
        nodes = []
        fail = []
        labels = []
        experiments = 6
        model = "knn"
        for n in range(experiments):
            self.consolidate(model,
                             mode_n=n,
                             model_offset=0,
                             subfolder=["experimental", "clean_sweep"])

            labels.append(self.logs[0]["experimental_mode"]["clean_d"])
            acc.append(self.get_dict_data("rrt")["accuracy"]["median"])
            nodes.append(self.get_list_data("rrt")["n_nodes"])
            fail.append(self.get_single_var_data("rrt")["fail_rate"])
            self.logs = []

        for key, clean_d in enumerate(labels):
            print("[d]: {} \t [acc]: {} \t [nodes]: {} \t [fail]: {} ".format(
                np.round(clean_d, 2),
                np.round(np.median(acc[key]), 2),
                np.median(nodes[key]),
                np.round(fail[key][0], 0)
            ))
        return None


def process(folder):
    print(folder)
    for model in ["clsgan", "knn"]:
        print("\n --------Evaluation for {}----------".format(model))
        p = Pipeline()
        if not p.consolidate(model, mode_n=-1, model_offset=0, subfolder=[folder]):
            continue
        p.print_mode()
        p.compute_model_stats()
        p.compute_rrt_stats()
        del p


if __name__ == "__main__":
    pass
    # Generate boxplots for sweeps
    # p = Pipeline()
    # p.reachability_box_plot()
    # p.reset()
    # p.clean_sweep_stats()
    # del p

    # Compute stats
    # for model in ["clsgan", "knn"]:
    #     print("\n --------Evaluation for {}----------".format(model))
    #     p = Pipeline()
    #     if not p.consolidate(model, mode_n=-1, model_offset=0, subfolder=["experimental", "2dof_final_multi"]):
    #         continue
    #     p.print_mode()
    #     p.compute_model_stats()
    #     p.compute_rrt_stats()
    #     del p
