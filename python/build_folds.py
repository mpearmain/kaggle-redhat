from sklearn.cross_validation import LabelKFold
import numpy as np
import pandas as pd

# Create 5-fold validation sets for model stacking to be used for all models.
# Can easily replicate for python modules, but R needs to use the same folds for stacking.

train = pd.read_csv("./input/act_train.csv",
                    usecols= ['people_id', 'activity_id'],
                    dtype={'people_id': np.str,
                           'activity_id': np.str}
                    )

kf = LabelKFold(train['people_id'], n_folds=5)
train.drop(['people_id'], inplace=True, axis=1)
train['fold5'] = kf.idxs

train.to_csv('./input/5-fold.csv', header=True, index=None)
