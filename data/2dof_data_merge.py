import numpy as np
import pandas as pd
import glob
import os

path = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(path, "merge", "*.csv")
files = glob.glob(pattern)

df_list = []
for _file in files:
    print("Loading {}".format(_file))
    df_list.append(pd.read_csv(_file))

df = pd.concat(df_list)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(os.path.join(path, "2dof_time.csv"), index=False)
print("Files successfully merged")
