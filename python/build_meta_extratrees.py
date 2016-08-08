# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score as auc
from python.binary_stacker import BinaryStackingClassifier
import datetime

if __name__ == '__main__':

    ## settings
    projPath = './'
    dataset_version = "v1"
    model_type = "etrees" 
    seed_value = 789
    todate = datetime.datetime.now().strftime("%Y%m%d")
    	    
    ## data
    # read the training and test sets HACK
    xtrain = pd.read_csv(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    id_train = xtrain.activity_id
    y_train = xtrain.outcome
    xtrain.drop('activity_id', axis = 1, inplace = True)
    xtrain.drop('outcome', axis = 1, inplace = True)

    xtest = pd.read_csv(projPath + 'input/xtest_ds_' + dataset_version + '.csv')
    id_test = xtest.activity_id
    xtest.drop('activity_id', axis = 1, inplace = True)

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
    
    ## model
    # setup model instances
    ets = ExtraTreesClassifier(criterion='gini',
                                 n_estimators=500,
                                 max_depth=None, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_leaf_nodes=None, 
                                 bootstrap=False,
                                 oob_score=False,
                                 n_jobs= -1,
                                 random_state= seed_value, 
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None)

    stacker = BinaryStackingClassifier([ets], xfolds=xfolds, evaluation=auc)
    stacker.fit(xtrain, y_train)



    # save the files            
    mvalid.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    mfull.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)
    