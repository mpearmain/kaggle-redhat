from __future__ import division
from __future__ import print_function

__author__ = 'michael.pearmain'

import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as auc
from bayes_opt import BayesianOptimization


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
              gamma,
              min_child_weight,
              silent=True,
              nthread=-1,
              seed=1234):

    clf = XGBClassifier(max_depth=int(max_depth),
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        silent=silent,
                        nthread=nthread,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        min_child_weight = min_child_weight,
                        seed=seed,
                        objective="binary:logistic")

    clf.fit(x0, y0, eval_metric="auc", eval_set=[(x1, y1)],verbose=False)
    ll = auc(y1, clf.predict_proba(x1)[:,1])
    return ll

if __name__ == "__main__":

    projPath = './'
    dataset_version = "v1"
    model_type = "xgb"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    xtrain = pd.read_csv(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    id_train = xtrain.activity_id
    y_train = xtrain.outcome
    xtrain.drop('activity_id', axis = 1, inplace = True)
    xtrain.drop('outcome', axis = 1, inplace = True)

    xtrain.rename(columns=lambda x: x.replace('_', ''), inplace=True)
    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')

    # work with validation split
    idx0 = xfolds[xfolds.fold5 != 1].index
    idx1 = xfolds[xfolds.fold5 == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = y_train[y_train.index.isin(idx0)]
    y1 = y_train[y_train.index.isin(idx1)]

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(6), int(20)),
                                      'learning_rate': (0.005, 0.4),
                                      'n_estimators': (int(500), int(1500)),
                                      'subsample': (0.6, 0.9),
                                      'colsample_bytree': (0.6, 0.9),
                                      'gamma': (0.0, 0.1),
                                      'min_child_weight': (int(1), int(10))
                                     })

    xgboostBO.maximize(init_points=5, n_iter=50, xi=0.05)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
    print('XGBOOST: %s' % xgboostBO.res['max']['max_params'])
