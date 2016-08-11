# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score as auc
from python.binary_stacker import BinaryStackingClassifier
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    datasets = ["v1", "v2"]
    model_type = "etrees"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    for dataset_version in datasets:
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

        ## model
        # setup model instances
        ets1 = ExtraTreesClassifier(criterion='gini', n_estimators=1000, n_jobs=-1, random_state=seed_value)
        ets2 = ExtraTreesClassifier(criterion='entropy', n_estimators=1000, n_jobs=-1, random_state=seed_value)

        stacker = BinaryStackingClassifier([ets1, ets2], xfolds=xfolds, evaluation=auc)
        stacker.colnames = ['ets1', 'ets2']
        stacker.fit(train, y_train)

        meta = stacker.meta_train
        meta['activity_id'] = id_train = train.activity_id
        meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)

        preds = stacker.predict_proba(test)
        preds['activity_id'] = id_test
        preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index=False, header=True)
