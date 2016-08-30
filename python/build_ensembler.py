# -*- coding: utf-8 -*-

import pandas as pd
from python.binary_stacker import BinaryStackingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score as auc
import xgboost as xgb
import datetime


if __name__ == '__main__':

    ## settings
    projPath = './'
    model_type = "ensembler"
    seed_value = 734
    todate = datetime.datetime.now().strftime("%Y%m%d")

    train = pd.read_csv(projPath + 'input/xvalid_20160821.csv')
    # train2 = pd.read_csv(projPath + 'input/prval2.csv')
    # train = pd.merge(train1, train2, left_on='activity_id', right_on='activity_id')
    # del train1, train2
    id_train = train.activity_id
    y_train = train.outcome
    train.drop('activity_id', axis = 1, inplace = True)
    train.drop('outcome', axis = 1, inplace = True)

    test = pd.read_csv(projPath + 'input/xfull_20160821.csv')
    # test2 = pd.read_csv(projPath + 'input/prfull.csv')
    # test = pd.merge(test1, test2, left_on='activity_id', right_on='activity_id')
    # del test1, test2

    id_test = test.activity_id
    test.drop('activity_id', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')

    train.rename(columns=lambda x: x.replace('_', ''), inplace=True)
    test.rename(columns=lambda x: x.replace('_', ''), inplace=True)

    classifiers = [
        AdaBoostClassifier(n_estimators=100, random_state=1234),
        AdaBoostClassifier(base_estimator=RidgeClassifier,n_estimators=100, random_state=1234, algorithm='SAMME'),
        xgb.XGBClassifier(n_estimators=400, nthread=-1, max_depth=12, min_child_weight=2.1, learning_rate=0.1,
                             silent=True, subsample=0.8, colsample_bytree=0.85, gamma=0.000000001, seed=1234),
        LogisticRegression(random_state=1234, n_jobs=-1),
        GaussianNB()
    ]

    stacker = BinaryStackingClassifier(classifiers, xfolds=xfolds, evaluation=auc)
    stacker.fit(train, y_train)



    meta = stacker.meta_train
    meta['activity_id'] = id_train
    meta['outcome'] = y_train
    meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset + '_seed' + str(
        seed_value) + '.csv', index=False, header=True)

    preds = stacker.predict_proba(test)
    preds['activity_id'] = id_test
    preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset + '_seed' + str(
        seed_value) + '.csv', index=False, header=True)