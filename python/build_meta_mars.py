# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from pyearth.earth import Earth
from sklearn.metrics import roc_auc_score as auc
from python.binary_stacker import BinaryStackingClassifier
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    datasets = ["v2"]
    model_type = "mars"
    seed_value = 78543
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
        # Combine Earth with LogisticRegression in a pipeline to do classification
        earth_classifier1 = Pipeline([('earth', Earth(max_degree=2, penalty=1.5)),
                                     ('logistic', LogisticRegression())])
        # Combine Earth with LogisticRegression in a pipeline to do classification
        earth_classifier2 = Pipeline([('earth', Earth(max_degree=3, penalty=2)),
                                     ('logistic', LogisticRegression())])

        stacker = BinaryStackingClassifier([earth_classifier1, earth_classifier2], xfolds=xfolds, evaluation=auc)
        stacker.fit(train, y_train)

        meta = stacker.meta_train
        meta['activity_id'] = id_train
        meta['outcome'] = y_train
        meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)

        preds = stacker.predict_proba(test)
        preds['activity_id'] = id_test
        preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index=False, header=True)
