import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def read_ds(path):
  """
  parse CSV data set and
  returns a tuple (input, target)
  """
  data = pd.read_csv(path, dtype=object, delimiter=",", header=None, skiprows=1, names=["id", "INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "INPUT_4", "INPUT_5", "INPUT_6", "INPUT_7", "INPUT_8", "INPUT_9", "TARGET_x", "TARGET_y", "TARGET_z"])
  y = data.drop(["id","INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "INPUT_4", "INPUT_5", "INPUT_6", "INPUT_7", "INPUT_8", "INPUT_9"], axis=1)
  X = data.drop(["id","TARGET_x", "TARGET_y", "TARGET_z"], axis=1).astype(float).to_numpy()

  y = y.astype(float).to_numpy()

  return (X , y)

CV = KFold(n_splits=5, random_state=42, shuffle=True)

### Initialization seed
SEED = 42