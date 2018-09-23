import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm
from timeit import default_timer as timer


class Cleaner():

    def __init__(self, threshold=0.15, percentage=0.2):
        self.d = threshold
        self.percentage = percentage
        pass

    def build_remove_list(self, labels):
        n = labels.shape[0]
        tree = KDTree(labels, leaf_size=10)
        remove_list = set()  # use set as hashtable to prevent duplicates
        for _ in tqdm(range(int(self.percentage*n))):
            idx = np.random.randint(0, n)
            subset = labels[idx, :].reshape(-1, 1).T
            distance, idx = tree.query(subset, k=2)

            if distance[0][1] < self.d:
                remove_list.add(idx[0][1])

        return remove_list

    def clean(self, data, labels):
        print("Start data cleaning process. Threshold: {}".format(self.d))
        start_points = data.shape[0]
        start = timer()
        while True:
            remove_list = self.build_remove_list(labels)
            if not len(remove_list):
                print("Cleaned in {} seconds".format(timer() - start))
                print("Removed {}%".format((start_points-data.shape[0])/start_points*100))
                return data, labels
            else:
                data = np.delete(data, list(remove_list), axis=0)
                labels = np.delete(labels, list(remove_list), axis=0)
                print("Removed {} datapoints".format(len(remove_list)))


class CleanAuto():
    def __init__(self, d=0.4):
        cleaner = Cleaner(threshold=d)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_dir, "pendulum_{}.csv".format('time'))
        df = pd.read_csv(filename)
        data = np.array(df)
        clean_data, clean_labels = cleaner.clean(data[:, :4], data[:, 4:])

        total_clean = np.hstack([clean_data, clean_labels])
        np.savetxt(
            os.path.join(current_dir, 'pendulum_time_clean.csv'),
            total_clean,
            delimiter=',',
            fmt="%f",
            header="lambda0, mu0, t_f, cost, theta0, omega0, theta1, omega1",
            comments="")


if __name__ == "__main__":
    clean = CleanAuto()
