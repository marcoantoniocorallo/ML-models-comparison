import pandas as pd
from sklearn.model_selection import StratifiedKFold

def read_ds(path):
  """
  parse CSV data set and
  returns a tuple (input, target)
  """

  df = pd.read_csv(path, sep=" ", names=['NaN','y','x1','x2','x3','x4','x5','x6','garbage'])
  y, df = df['y'], df.drop(columns=['NaN','garbage','y'])
  
  # One-hot encoding categorical variables
  df = pd.get_dummies(df, columns=['x1','x2','x3','x4','x5','x6']).astype('int')
  return (df, y)

### Global K-Fold strategy
CV = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

### Initialization seed
SEED = 42