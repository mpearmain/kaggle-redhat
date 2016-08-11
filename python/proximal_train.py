import pyximport
pyximport.install()

import numpy as np
import glob, os

from scipy.sparse import vstack

from sklearn.metrics import mean_squared_error

from proximalFM.proximal_fm import ProximalFM
from proximalFM.data_reader import DataReader


def rmsle(act, preds):
    return np.sqrt(mean_squared_error( np.log1p(act), np.log1p(preds)))


if __name__ == "__main__":
    FMT = '%H:%M:%S'
    # Construct loop around the weeks and always check against validation.
    os.chdir("../input/")
    weeklist = []
    week_files = sorted(glob.glob("xtrain*[3-9].csv"))
    print week_files
    os.chdir("../python/")

    config =  {"model": {
        "modelFile": "model.pkl",
        "alpha": 0.06,  # learning rate
        "beta": 1,  # smoothing parameter for adaptive learning rate
        "L1": 0.07,  # L1 regularization, larger value means more regularized
        "L2": 0.03,  # L2 regularization, larger value means more regularized
        "enable_fm": True,  # include FM component
        "alpha_fm": 0.0593,  # FM learning rate
        "beta_fm": 1.,  # FM smoothing parameter for adaptive learning rate
        "L1_fm": 0.2,  # FM L1 regularization
        "L2_fm": 0.8,  # FM L2 regularization
        "fm_dim": 7,  # number of factors for feature interactions
        "fm_initDev": 0.2,  # standard deviation for random initialisation of FM weights
        "D": 2 ** 25,
        "report_frequency": 100000,
        "warm_start": False,  # if True, reuse the solution of the previous call to fit otherwise erase
        # previous solution
        "epoch": 5  # learn training data for N passes
    },
        "data_dictionary": {
            "label": 'AdjDemand',
            "header": ['id', 'DepotID', 'ChannelID', 'RouteID', 'ClientID', 'ProductID', 'AdjDemand',
                       'WeeklyProductCounts', 'WeeklyClientCounts', 'WeeklyDepotIDCounts', 'WeeklyChannelCounts',
                       'Town_En_Counts', 'State_En_Counts', 'Towns_in_State', 'ClientNameLogCounts', 'inches', 'weight',
                       'liquids', 'pieces', 'has_vanilla', 'has_choco', 'has_tortil',
                       'has_hotdog', 'has_energy'],
            "features": ['DepotID', 'ChannelID', 'RouteID', 'ClientID', 'ProductID',
                         'WeeklyProductCounts', 'WeeklyClientCounts', 'WeeklyDepotIDCounts', 'WeeklyChannelCounts',
                         'Town_En_Counts', 'State_En_Counts', 'Towns_in_State', 'ClientNameLogCounts', 'inches', 'weight',
                         'liquids', 'pieces', 'has_vanilla', 'has_choco', 'has_tortil',
                         'has_hotdog', 'has_energy'],
            "features_dim": 25,
        }
    }

    ##################### Week 3 #############################
    reader = DataReader(config['data_dictionary'])
    learner = ProximalFM(config['model'])
    for week in week_files:
        print "Loading Week file", week
        X_train, y_train = reader.load_data("../input/"+week)
        Xn_train = reader.transform(X_train)

        del X_train

        y_train = y_train.astype('int32')
        y_train = np.log1p(y_train)
        print "Training"
        learner.fit(Xn_train, y_train)
        del Xn_train, y_train

        if week == 'xtrain_week7.csv':
            print "Building out validation week 8"
            X_valid, y_valid = reader.load_data("../input/xtrain_week8.csv")
            Xn_valid = reader.transform(X_valid)
            del X_valid
            y_valid = y_valid.astype('int32')
            y_valid = np.log1p(y_valid)
            preds = np.expm1(learner.predict_proba(Xn_valid))
            preds[preds < 0] = 0
            loss = rmsle(np.expm1(y_valid), preds)
            del y_valid, Xn_valid
            print "TRAIN WEEKS 3-7, VALIDATION 8 LOSS:", loss
            np.savetxt("../input/pr_val_week8_proximalFM.csv", preds, delimiter=",")

            print "Building out validation"
            X_valid, y_valid = reader.load_data("../input/xtrain_week9.csv")
            Xn_valid = reader.transform(X_valid)
            del X_valid
            y_valid = y_valid.astype('int32')
            y_valid = np.log1p(y_valid)
            preds = np.expm1(learner.predict_proba(Xn_valid))
            preds[preds < 0] = 0
            loss = rmsle(np.expm1(y_valid), preds)
            del y_valid, Xn_valid
            print "TRAIN WEEKS 3-7, VALIDATION 9 LOSS:", loss
            np.savetxt("../input/pr_val_week9_proximalFM.csv", preds, delimiter=",")

    #### TEST #######
    X_test, y_test = reader.load_data("../input/xtest.csv")
    Xn_test = reader.transform(X_test)
    del X_test
    y_test = y_test.astype('int32')
    y_test = np.log1p(y_test)
    preds = np.expm1(learner.predict_proba(Xn_test))
    preds[preds < 0] = 0
    np.savetxt("../input/pr_full_proximalFM.csv", preds, delimiter=",")


