
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc


if __name__ == "__main__":

    projPath = './'
    model_type = "xgb_hash_bag"
    seed_value = 674839
    todate = datetime.datetime.now().strftime("%Y%m%d")

    dtrain = xgb.DMatrix(projPath + 'input/dtrain.data')
    dtest = xgb.DMatrix(projPath + 'input/dtest.data')

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv', dtype={'fold5': 'int32'})
    # work with 5-fold split
    fold_index = np.array(xfolds.fold5)
    n_folds = len(np.unique(fold_index))

    # storage structure for forecasts
    mvalid = np.zeros((dtrain.num_row(),1))
    mfull = np.zeros((dtest.num_row(),1))

    nbag =5

    # loop over folds
    # Recompile model on each fold
    for j in range(0,n_folds):
        # configure model with j-th combo of parameters
        idx0 = np.where(fold_index != j)[0].tolist()
        idx1 = np.where(fold_index == j)[0].tolist()
        x0 = dtrain.slice(idx0)
        x1 = dtrain.slice(idx1)

        # setup bagging classifier
        pred_sum = 0
        for k in range(nbag):
            seed = 1+(nbag*100)+ (nbag^2)
            print 'Fold:', j, 'Building bag:', k
            params = {
                "objective": "binary:logistic",
                "booster": "gblinear",
                "eval_metric": "auc",
                "eta": 0.101,
                "tree_method": 'exact',
                "max_depth": 13,
                "min_child_weight": 0,
                "gamma": 0.0001,
                "subsample": 0.76,
                "colsample_bytree": 0.8,
                "silent": True,
                "seed": seed,
            }
            num_boost_round = 1408
            #num_boost_round = 4
            gbm = xgb.train(params, x0, num_boost_round)
            preds = gbm.predict(x1)
            pred_sum += preds
            pred_average = pred_sum / (k+1)
            print 'AUC val:', auc(x1.get_label(), pred_average)
            print 'Finished bag:', k
        mvalid[idx1,0] = pred_average
        print "finished fold:", j

    print "Building full prediction model for test set."
    # configure model with j-th combo of parameters

    # setup bagging classifier
    pred_sum = 0
    for k in range(nbag):
        seed = 1 + (nbag * 100) + (nbag ^ 2)
        print 'Fold:', j, 'Building bag:', k
        params = {
            "objective": "binary:logistic",
            "booster": "gblinear",
            "eval_metric": "auc",
            "eta": 0.101,
            "tree_method": 'exact',
            "max_depth": 13,
            "min_child_weight": 0,
            "gamma": 0.0001,
            "subsample": 0.76,
            "colsample_bytree": 0.8,
            "silent": True,
            "seed": seed,
        }
        num_boost_round = 1408
        gbm = xgb.train(params, dtrain, num_boost_round)
        preds = gbm.predict(dtest)
        pred_sum += preds
        pred_average = pred_sum / (k + 1)
        print 'Finished bag:', k
    mfull[:,0] = pred_average
    print "finished full prediction"

    ## store the results
    # add indices etc
    mvalid = pd.DataFrame(mvalid)
    mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
    train = pd.read_csv("./input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    mvalid['activity_id'] = train['activity_id']
    del train

    mfull = pd.DataFrame(mfull)
    mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
    print("Load test.csv...")
    test = pd.read_csv("./input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])
    id_test = test['activity_id']
    del test
    mfull['activity_id'] = id_test

    # save the files
    mvalid.to_csv('./metafeatures/prval_' + model_type + '_' + todate + '_data' + model_type + '.csv', index = False, header = True)
    mfull.to_csv('./metafeatures/prfull_' + model_type + '_' + todate + '_data' + model_type + '.csv', index = False, header = True)
