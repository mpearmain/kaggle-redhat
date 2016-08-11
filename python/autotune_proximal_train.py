import pyximport
pyximport.install()

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from bayes_opt import BayesianOptimization

from proximalFM.proximal_fm import ProximalFM
from proximalFM.data_reader import DataReader


def proximal_bayes(alpha, L1, L2, alpha_fm, L1_fm, L2_fm, fm_dim, fm_initDev, epoch):
    # setup bagging classifier
    config = {"model": {
        "modelFile": "model.pkl",
        "alpha": alpha,  # learning rate
        "beta": 1,  # smoothing parameter for adaptive learning rate
        "L1": L1,  # L1 regularization, larger value means more regularized
        "L2": L2,  # L2 regularization, larger value means more regularized
        "enable_fm": True,  # include FM component
        "alpha_fm": alpha_fm,  # FM learning rate
        "beta_fm": 1.,  # FM smoothing parameter for adaptive learning rate
        "L1_fm": L1_fm,  # FM L1 regularization
        "L2_fm": L2_fm,  # FM L2 regularization
        "fm_dim": fm_dim,  # number of factors for feature interactions
        "fm_initDev": fm_initDev,  # standard deviation for random initialisation of FM weights
        "D": 2 ** 25,
        "report_frequency": 100000,
        "warm_start": False,  # if True, reuse the solution of the previous call to fit otherwise erase
        # previous solution
        "epoch": epoch  # learn training data for N passes
    }
    }

    learner = ProximalFM(config['model'])
    learner.fit(Xn_train, y0)

    preds = learner.predict_proba(Xn_valid)[:, 1]
    loss = auc(y1, preds)
    return loss


if __name__ == "__main__":

    projPath = './'
    dataset_version = "v1"
    model_type = "proximalFM"
    seed_value = 789775
    configR = {"data_dictionary": {
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
                     'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'outcome', 'tyear', 'tmonth',
                     'tyearweek', 'tday', 't_sum_true', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y',
                     'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
                     'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
                     'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28',
                     'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
                     'char_37', 'char_38', 'p_sum_true', 'pyear', 'pmonth', 'pyearweek', 'pday', 'days_diff'],
        "features_dim": 25,
    }
    }


    reader = DataReader(configR['data_dictionary'])
    ## data
    xtrain, ytrain = reader.load_data(projPath + 'input/xtrain_ds_' + dataset_version + '.csv')
    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
    fold_index = xfolds.fold5

    # work with validation split
    print "Creating validation for bayes tuning."
    idx0 = np.where(fold_index != 0)
    idx1 = np.where(fold_index == 0)
    x0 = np.array(xtrain)[idx0, :][0]
    x1 = np.array(xtrain)[idx1, :][0]
    y0 = np.array(ytrain)[idx0]
    y1 = np.array(ytrain)[idx1]

    Xn_train = reader.transform(x0)
    Xn_valid = reader.transform(x1)

    proximal_bayes = BayesianOptimization(proximal_bayes,
                                          {"alpha": (0.001, 0.5),
                                           "L1": (0., 2.),
                                           "L2": (0., 2.),
                                           "alpha_fm": (0.0001, 0.5),
                                           "L1_fm": (0., 2.),
                                           "L2_fm": (0., 2.),
                                           "fm_dim": (int(4), int(15)),
                                           "fm_initDev": (0.05, 0.3),
                                           "epoch": (int(2), int(20))
                                           })
    proximal_bayes.maximize(init_points=5, n_iter=50)
    print('-' * 53)

    print('Final Results')
    print('PromixmalFM: %f' % proximal_bayes.res['max']['max_val'])
    print(proximal_bayes.res['max']['max_params'])
