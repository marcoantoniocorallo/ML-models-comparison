import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# read CSV dataset
def read_ds(path):
  """
  parse CSV data set and
  returns a tuple (input, target)
  """
  names = ["id", "INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "INPUT_4", "INPUT_5", "INPUT_6", "INPUT_7", "INPUT_8", "INPUT_9", "TARGET_x", "TARGET_y", "TARGET_z"]
  data = pd.read_csv(path, dtype=object, delimiter=",", header=None, skiprows=1, names=names)
  y = data.drop(["id","INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "INPUT_4", "INPUT_5", "INPUT_6", "INPUT_7", "INPUT_8", "INPUT_9"], axis=1)
  X = data.drop(["id","TARGET_x", "TARGET_y", "TARGET_z"], axis=1).astype(float).to_numpy()

  y = y.astype(float).to_numpy()

  return (X , y)

# read CSV blind test-set
def read_ts(path):
  names = ['id','input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9']
  df = pd.read_csv(path, names=names, dtype=object, header=None, skipinitialspace=True, skiprows=7)
  return df.drop(['id'],axis=1).astype(float).to_numpy()

# Initialization seed
SEED = 42

# kfolding strategy
CV = KFold(n_splits=5, random_state=SEED, shuffle=True)