# -*- coding: utf-8 -*-

import pandas as pd
import datetime
from python.binary_stacker import BinaryStackingClassifier
from sklearn.metrics import roc_auc_score as auc
import xgboost as xgb


if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "v1"
    model_type = "xgb"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    train = pd.read_csv(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    id_train = train.activity_id
    y_train = train.outcome
    train.drop('activity_id', axis = 1, inplace = True)
    train.drop('outcome', axis = 1, inplace = True)

    test = pd.read_csv(projPath + 'input/xtest_ds_' + dataset_version + '.csv')
    id_test = test.activity_id
    test.drop('activity_id', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')

    train.rename(columns=lambda x: x.replace('_', ''), inplace=True)
    test.rename(columns=lambda x: x.replace('_', ''), inplace=True)

    bst1 = xgb.XGBClassifier(n_estimators=1364, nthread=-1, max_depth=6, min_child_weight=1.0, learning_rate=0.4,
                             silent=True, subsample=0.9, colsample_bytree=0.6, gamma=0.05, seed=seed_value)
    bst2 = xgb.XGBClassifier(n_estimators=1364, nthread=-1, max_depth=6, min_child_weight=1.0, learning_rate=0.4,
                             silent=True, subsample=0.9, colsample_bytree=0.6, gamma=0.05, seed=seed_value,)
    bst3 = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=12, min_child_weight=1.0, learning_rate=0.05,
                             silent=True, subsample=0.85, colsample_bytree=0.78, gamma=0.0000001, seed=seed_value)
    bst4 = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=12, min_child_weight=1.0, learning_rate=0.05,
                             silent=True, subsample=0.85, colsample_bytree=0.78, gamma=0.0000001, seed=seed_value)
    bst5 = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=12, min_child_weight=1.0, learning_rate=0.05,
                             silent=True, subsample=0.85, colsample_bytree=0.78, gamma=0.0000001, seed=seed_value)
    bst6 = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=12, min_child_weight=1.0, learning_rate=0.05,
                             silent=True, subsample=0.85, colsample_bytree=0.78, gamma=0.0000001, seed=seed_value)
    

    stacker = BinaryStackingClassifier([bst1, bst2, bst3, bst4, bst5, bst6], xfolds=xfolds, evaluation=auc)
    stacker.colnames = ['bst1', 'bst2', 'bst3', 'bst4', 'bst5', 'bst6']
    stacker.fit(train, y_train, eval_metric='auc')

    meta = stacker.meta_train
    meta['activity_id'] = train['activity_id']
    meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(
        seed_value) + '.csv', index=False, header=True)
    
    preds = stacker.predict_proba(test)
    preds['activity_id'] = id_test
    preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(
        seed_value) + '.csv', index=False, header=True)
