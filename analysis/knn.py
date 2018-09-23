import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.cleaner import CleanAuto
from data.loader import loader as load
from models import KNN
from trainers import KNNTrainer
from settings import settings

cfg = settings()
clean_d = [0.01, 0.05, 0.1, 0.15, 0.3, 0.6]
# clean_d = [0.01, 0.05]
model = KNN(**cfg.model["knn"].structure)
trainer = KNNTrainer(model)
df = pd.DataFrame()
for _, d in enumerate(clean_d):
    print("Cleaning d: {}".format(d))
    c = CleanAuto(d)
    trainer.train(load.training_data, load.training_labels)
    model_trained = trainer.get_model()
    costates_hat = model_trained.predict(load.test_labels)
    p = np.linalg.norm(costates_hat[:, :2], axis=1)
    df = pd.concat([df, pd.DataFrame({d: p})], axis=1, ignore_index=True)

# Rename columns
df.columns = clean_d

df.plot(kind="box")
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "knn_boxplot.png")
plt.savefig(filename)
