from sklearn.cross_validation import LabelKFold
import numpy as np
import pandas as pd

# Create 5-fold validation sets for model stacking to be used for all models.
# Can easily replicate for python modules, but R needs to use the same folds for stacking.

train = pd.read_csv("./input/act_train.csv",
                    dtype={'people_id': np.str,
                           'activity_id': np.str,
                           'outcome': np.int8},
                    parse_dates=['date'])

kf = LabelKFold(train['people_id'], n_folds=5)
np.savetxt("./input/5-fold.csv", kf.idxs, delimiter=",")
