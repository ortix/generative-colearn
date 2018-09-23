import os

import numpy as np
import pandas as pd
from sklearn import preprocessing

from settings import settings as cfg
from data.cleaner import Cleaner


# Class is instantiated at the bottom!
class DataLoader():

    def __init__(self, load=True):
        self.training_data = []
        self.training_labels = []
        self.test_data = []
        self.test_labels = []
        self.cfg = cfg()
        self.clean_available = False
        self.keys = []
        self.transformer = []
        self.is_loaded = False
        if load:
            self.init_load()
        return None

    def init_load(self, filename=None, force=False):
        if not self.is_loaded or force:
            df = self._load(filename)
            data, labels = self._pre_process(df)
            data, labels = self._clean(data, labels)
            self._split(data, labels)
            self.is_loaded = True
        return None

    def next_batch(self, batch_size, test=False):
        '''
        Return a total of `num` random samples and labels. 
        '''
        if test:
            data = self.test_data
            labels = self.test_labels
        else:
            data = self.training_data
            labels = self.training_labels

        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        # data_shuffle = [data[i] for i in idx]
        # labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data[idx]), np.asarray(labels[idx])

    def set_training_data(self, data):
        self.training_data = data

    def set_training_labels(self, labels):
        self.training_labels = labels

    def training_set(self):
        '''
        Returns a list of training data and labels
        '''
        return self.training_data, self.training_labels

    def test_set(self):
        '''
        Returns a list of test data and labels
        '''
        return self.test_data, self.test_labels

    def _clean(self, data, labels):
        # if not self.clean_available and not self.cfg.model.load and self.cfg.model.clean:
        if not self.cfg.model.clean:
            return data, labels
        elif self.cfg.simulation.load and self.clean_available:
            return data, labels

        cleaner = Cleaner(**self.cfg.cleaner)
        data, labels = cleaner.clean(data, labels)
        self._save_clean(data, labels)
        self.clean_available = True

        return data, labels

    def _pre_process(self, df):

        # We use these marginalized keys to group states together
        keys = ["theta0", "omega0", "theta1", "omega1", "lambda", "mu", "t_f", "cost"]

        df = {key: np.array(df.filter(like=key)) for key in keys}

        data = np.hstack([df["lambda"], df["mu"], df["t_f"], df["cost"]])
        labels = np.hstack([df["theta0"], df["omega0"], df["theta1"], df["omega1"]])

        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(-1,1))
        # scaler.fit(labels)
        # labels = scaler.transform(labels)
        # self.transformer = scaler
        return data, labels

    def _save_clean(self, data, labels, clean=False):
        # Quick and dirty key generation
        n_states = int(labels.shape[1]/4)
        m_keys = ["theta0", "omega0", "theta1", "omega1", "lambda", "mu"]
        key_list = []
        for _, key in enumerate(m_keys):
            key_list.append(["{}{}".format(key, i) for i in range(1, n_states+1)])
        keys = np.hstack([np.array(key_list).flat[:], ["cost", "t_f"]])
        keys_str = ", ".join(keys)
        # print(keys_str)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_dir, "{}_{}_clean.csv".format(
            self.cfg.simulation.system, self.cfg.simulation.mode))
        np.savetxt(
            filename,
            np.hstack([labels, data]),
            delimiter=',',
            header=keys_str,
            comments="",
            fmt="%f")
        pass

    def _load(self, filename_override=None):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(current_dir, "{}_{}.csv".format(
            self.cfg.simulation.system, self.cfg.simulation.mode))

        filename_clean = os.path.join(current_dir, "{}_{}_clean.csv".format(
            self.cfg.simulation.system, self.cfg.simulation.mode))
        self.clean_available = os.path.isfile(filename_clean)

        if self.cfg.model.clean and self.clean_available and self.cfg.simulation.load:
            filename = filename_clean
        if filename_override is not None:
            print("Filename override: {}".format(filename_override))
            filename = os.path.join(current_dir, filename_override)
        try:
            df = pd.read_csv(filename)
            print("Loaded {}".format(filename))
        except FileNotFoundError:
            raise ValueError("Could not load {}. Going to generate it!".format(filename))
        else:
            return df

    def _split(self, data, labels):

        split_n = int(self.cfg.simulation.split*data.shape[0])
        self.training_data = data[split_n:, :]
        self.training_labels = labels[split_n:, :]

        self.test_data = data[:split_n, :]
        self.test_labels = labels[:split_n, :]


loader = DataLoader(False)
