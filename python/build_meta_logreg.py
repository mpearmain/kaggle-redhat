# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from python.binary_stacker import BinaryStackingClassifier
from sklearn.metrics import roc_auc_score as auc
import datetime


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName).mean().reset_index()
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if (useLOO):
        x = ((x * x.shape[0]) - outcomes) / (x.shape[0] - 1)
    return x.fillna(x.mean())


if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "v1"
    model_type = "LR"
    seed_value = 7875
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    train = pd.read_csv(projPath + 'input/xtrain_ds_' + dataset_version + '.csv',
                        usecols=['people_id',
                                 'outcome',
                                 'activity_id',
                                 'group_1',
                                 'char_2_y',
                                 'char_38',
                                 'p_sum_true',
                                 'days_diff'])
    test = pd.read_csv(projPath + 'input/xtest_ds_' + dataset_version + '.csv',
                       usecols=['people_id',
                                'activity_id',
                                'group_1',
                                'char_2_y',
                                'char_38',
                                'p_sum_true',
                                'days_diff'])
    test['outcome'] = 0

    lootrain = pd.DataFrame()
    for col in train.columns:
        if (col != 'outcome' and col != 'people_id' and col != 'activity_id'):
            print(col)
            lootrain[col] = LeaveOneOut(train, train, col, True).values
    lootrain['activity_id'] = train['activity_id']

    lootest = pd.DataFrame()

    for col in train.columns:
        if (col != 'outcome' and col != 'people_id'):
            print(col)
            lootest[col] = LeaveOneOut(train, test, col, False).values

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
    
    ## model
    # setup model instances
    lr1 = LogisticRegression(penalty='l2', C= 100000.0, n_jobs=-1, random_state=seed_value)
    lr2 = LogisticRegression(penalty='l1', C= 1000.0, n_jobs=-1, random_state=seed_value)
    lr3 = LogisticRegression(penalty='l2', C= .6, n_jobs=-1, random_state=seed_value)
    lr4 = LogisticRegression(penalty='l1', C= 6, n_jobs=-1, random_state=seed_value)

    stacker = BinaryStackingClassifier([lr1, lr2, lr3, lr4], xfolds=xfolds, evaluation=auc)
    stacker.colnames = ['lr1', 'lr2', 'lr3', 'lr4']
    stacker.fit(lootrain[['group_1', 'char_2_y', 'char_38', 'p_sum_true', 'days_diff']], train['outcome'])

    meta = stacker.meta_train
    meta['activity_id'] = train['activity_id']
    meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)

    preds = stacker.predict_proba(lootest[['group_1', 'char_2_y', 'char_38', 'p_sum_true', 'days_diff']])
    preds['activity_id'] = test['activity_id']
    preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index=False, header=True)
