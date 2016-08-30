from __future__ import division
from __future__ import print_function

__author__ = 'michael.pearmain'

import pandas as pd
import datetime
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from pyearth.earth import Earth
from sklearn.metrics import roc_auc_score as auc
from bayes_opt import BayesianOptimization


def mars_tune(max_degree, penalty):
    # Combine Earth with LogisticRegression in a pipeline to do classification
    clf = Pipeline([('earth', Earth(max_degree=int(max_degree), penalty=penalty)),
                             ('logistic', LogisticRegression())])

    clf.fit(x0, y0)
    ll = auc(y1, clf.predict_proba(x1)[:,1])
    return ll

if __name__ == "__main__":

    projPath = './'
    dataset_version = "v2"
    model_type = "mars"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    xtrain = pd.read_csv(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    id_train = xtrain.activity_id
    y_train = xtrain.outcome
    xtrain.drop('activity_id', axis = 1, inplace = True)
    xtrain.drop('outcome', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')

    # work with validation split
    idx0 = xfolds[xfolds.fold5 != 1].index
    idx1 = xfolds[xfolds.fold5 == 1].index
    x0 = xtrain[xtrain.index.isin(idx0)]
    x1 = xtrain[xtrain.index.isin(idx1)]
    y0 = y_train[y_train.index.isin(idx0)]
    y1 = y_train[y_train.index.isin(idx1)]

    marsBO = BayesianOptimization(mars_tune,
                                     {'max_degree': (int(1), int(5)),
                                      'penalty': (0., 10.),
                                     })

    marsBO.maximize(init_points=5, n_iter=10, xi=0.05)
    print('-' * 53)

    print('Final Results')
    print('Mars: %f' % marsBO.res['max']['max_val'])
    print('Mars: %s' % marsBO.res['max']['max_params'])
