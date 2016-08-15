import pyximport
pyximport.install()

import pandas as pd
import datetime

from sklearn.metrics import roc_auc_score as auc

from proximalFM.proximal_fm import ProximalFM
from proximalFM.data_reader import DataReader

from binary_stacker_np import BinaryStackingClassifierNP

if __name__ == "__main__":

    projPath = './'
    dataset_version = "v1"
    model_type = "proximalFM"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    train_config1 =  {"model": {
        "modelFile": "model.pkl",
        "alpha": 0.3072,  # learning rate
        "beta": 1,  # smoothing parameter for adaptive learning rate
        "L1": 1.12,  # L1 regularization, larger value means more regularized
        "L2": 0.1661,  # L2 regularization, larger value means more regularized
        "enable_fm": True,  # include FM component
        "alpha_fm": 0.03,  # FM learning rate
        "beta_fm": 1.,  # FM smoothing parameter for adaptive learning rate
        "L1_fm": 1.079,  # FM L1 regularization
        "L2_fm": 1.2789,  # FM L2 regularization
        "fm_dim": 8,  # number of factors for feature interactions
        "fm_initDev": 0.2311,  # standard deviation for random initialisation of FM weights
        "D": 2 ** 25,
        "report_frequency": 100000,
        "warm_start": False,  # if True, reuse the solution of the previous call to fit otherwise erase
        # previous solution
        "epoch": 8  # learn training data for N passes
    },
        "data_dictionary": {
        "label": 'outcome',
        "header": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                   'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'outcome', 'tyear', 'tmonth',
                   'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                   'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                   'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                   'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                   'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                   'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'activity_id', 'days_diff'],
        "features": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                     'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'tyear', 'tmonth',
                     'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                     'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                     'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                     'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                     'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                     'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'days_diff'],
        "features_dim": 25,
        }
    }

    train_config2 =  {"model": {
        "modelFile": "model.pkl",
        "alpha": 0.2,  # learning rate
        "beta": 1,  # smoothing parameter for adaptive learning rate
        "L1": 0.91,  # L1 regularization, larger value means more regularized
        "L2": 0.764,  # L2 regularization, larger value means more regularized
        "enable_fm": True,  # include FM component
        "alpha_fm": 0.104,  # FM learning rate
        "beta_fm": 1.,  # FM smoothing parameter for adaptive learning rate
        "L1_fm": 0.00134,  # FM L1 regularization
        "L2_fm": 1.07,  # FM L2 regularization
        "fm_dim": 4,  # number of factors for feature interactions
        "fm_initDev": 0.1036,  # standard deviation for random initialisation of FM weights
        "D": 2 ** 25,
        "report_frequency": 100000,
        "warm_start": False,  # if True, reuse the solution of the previous call to fit otherwise erase
        # previous solution
        "epoch": 5  # learn training data for N passes
    },
        "data_dictionary": {
        "label": 'outcome',
        "header": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                   'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'outcome', 'tyear', 'tmonth',
                   'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                   'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                   'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                   'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                   'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                   'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'activity_id', 'days_diff'],
        "features": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                     'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'tyear', 'tmonth',
                     'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                     'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                     'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                     'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                     'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                     'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'days_diff'],
        "features_dim": 25,
        }
    }


    test_config = {
        "data_dictionary": {
            "label": 'people_id', # Just a dummy holder
            "header": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                       'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'tyear', 'tmonth',
                       'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                       'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                       'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                       'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                       'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                       'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'activity_id',
                       'days_diff'],
            "features": ['people_id', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x',
                         'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'tyear', 'tmonth',
                         'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                         'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                         'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                         'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                         'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                         'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'days_diff'],
            "features_dim": 25,
        }
    }

    ##################### Week 3 #############################
    ## data
    train_reader = DataReader(train_config1['data_dictionary'])
    test_reader = DataReader(test_config['data_dictionary'])

    xtrain, ytrain = train_reader.load_data(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    ytrain = ytrain.astype('int32')
    xtest = test_reader.load_data(projPath + 'input/xtest_ds_' + dataset_version + '.csv', test_data=True)
    submission = pd.read_csv(projPath + 'input/sample_submission.csv')
    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
    fold_index = xfolds.fold5

    Xn_train = train_reader.transform(xtrain)
    Xn_test = test_reader.transform(xtest)

    ftrlfm1 = ProximalFM(train_config1['model'])
    ftrlfm2 = ProximalFM(train_config2['model'])

    stacker = BinaryStackingClassifierNP([ftrlfm2, ftrlfm1], xfolds=xfolds, evaluation=auc)
    stacker.fit(Xn_train, ytrain)

    meta = stacker.meta_train
    meta['activity_id'] = xfolds['activity_id']
    meta['outcome'] = ytrain
    meta.to_csv(projPath + 'metafeatures/prval_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index = False, header = True)

    preds = stacker.predict_proba(Xn_test)
    preds['activity_id'] = submission['activity_id']
    preds.to_csv(projPath + 'metafeatures/prfull_' + model_type + '_' + todate + '_data' + dataset_version + '_seed' + str(seed_value) + '.csv', index=False, header=True)
