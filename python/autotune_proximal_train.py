import pyximport
pyximport.install()

import numpy as np
import glob, os
from bayes_opt import BayesianOptimization


from sklearn.metrics import mean_squared_error

from proximalFM.proximal_fm import ProximalFM
from proximalFM.data_reader import DataReader


def rmsle(act, preds):
    return np.sqrt(mean_squared_error( np.log1p(act), np.log1p(preds)))

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
    learner.fit(Xn_train, y_train)

    preds = np.expm1(learner.predict_proba(Xn_test))
    preds[preds < 0] = 0
    loss = rmsle(np.expm1(y_test), preds)
    return loss


if __name__ == "__main__":
    FMT = '%H:%M:%S'
    # Construct loop around the weeks and always check against validation.
    os.chdir("../input/")
    weeklist = []
    week_files = sorted(glob.glob("xtrain*[3-9].csv"))
    print week_files
    os.chdir("../python/")

    configR = {"data_dictionary": {
        "label": 'AdjDemand',
        "header": ['id', 'WeekNum', 'DepotID', 'ChannelID', 'RouteID', 'ClientID', 'ProductID', 'AdjDemand',
                   'WeeklyProductCounts', 'WeeklyClientCounts', 'WeeklyDepotIDCounts', 'WeeklyChannelCounts',
                   'Town_En_Counts', 'State_En_Counts', 'Towns_in_State', 'ClientNameLogCounts', 'inches', 'weight',
                   'liquids', 'pieces', 'short_nameCounts', 'brandCounts', 'has_vanilla', 'has_choco', 'has_tortil',
                   'has_hotdog', 'has_energy', 'agu', 'aven', 'azuc', 'barr', 'barrit', 'bco', 'bes', 'bigot',
                   'bimboll', 'bk', 'blanc', 'boll', 'bols', 'bran', 'burrit', 'canapin', 'canel', 'canelit', 'chisp',
                   'choc', 'chochit', 'chocochisp', 'chocolat', 'clasic', 'conch', 'cong', 'cuernit', 'dalmat',
                   'delici', 'dobl', 'dogs', 'don', 'doradit', 'duo', 'escol', 'extra', 'fibr', 'fres', 'frut', 'fs',
                   'fsa', 'gallet', 'gansit', 'harin', 'hna', 'hot', 'integral', 'jamon', 'kc', 'lat', 'linaz', 'lors',
                   'maiz', 'mantec', 'manzan', 'mari', 'mas', 'medi', 'mg', 'milk', 'mini', 'mix', 'mm', 'mol',
                   'multigran', 'nit', 'noch', 'nuez', 'ondul', 'orejit', 'pack', 'pan', 'paner', 'panqu', 'pastiset',
                   'pin', 'pinguin', 'plativol', 'polvoron', 'princip', 'reban', 'ric', 'rock', 'rol', 'sal', 'salm',
                   'salv', 'sandwich', 'siluet', 'sponch', 'suavicrem', 'submarin', 'sup', 'surt', 'tartin', 'thins',
                   'tir', 'tortill', 'tortillin', 'tost', 'totop', 'trak', 'triki', 'tub', 'vainill', 'wond',
                   'zarzamor'],
        "features": ['WeekNum', 'DepotID', 'ChannelID', 'RouteID', 'ClientID', 'ProductID',
                     'WeeklyProductCounts', 'WeeklyClientCounts', 'WeeklyDepotIDCounts', 'WeeklyChannelCounts',
                     'Town_En_Counts', 'State_En_Counts', 'Towns_in_State', 'ClientNameLogCounts', 'inches', 'weight',
                     'liquids', 'pieces', 'short_nameCounts', 'brandCounts', 'has_vanilla', 'has_choco', 'has_tortil',
                     'has_hotdog', 'has_energy'],
        "features_dim": 25,
    }
    }

    ##################### Week 3 #############################
    ## Have to run week 3 independent to generate weights to be used
    reader = DataReader(configR['data_dictionary'])
    X_train, y_train = reader.load_data("../input/xtrain_week3.csv")
    Xn_train = reader.transform(X_train)
    del X_train
    y_train = y_train.astype('int32')
    y_train = np.log1p(y_train)

    #### TEST #######
    X_test, y_test = reader.load_data("../input/xtrain_week9.csv")
    Xn_test = reader.transform(X_test)
    del X_test
    y_test = y_test.astype('int32')
    y_test = np.log1p(y_test)

    proximal_bayes = BayesianOptimization(proximal_bayes,
                                          {"alpha": (0.01, 0.1),
                                           "L1": (0., 1.),
                                           "L2": (0., 2.),
                                           "alpha_fm": (0.001, 0.2),
                                           "L1_fm": (0., 1.),
                                           "L2_fm": (0., 2.),
                                           "fm_dim": (int(4), int(10)),
                                           "fm_initDev": (0.05, 0.3),
                                           "epoch": (int(2), int(10))
                                           })
    proximal_bayes.maximize(init_points=5, n_iter=50)
    print('-' * 53)

    print('Final Results')
    print('PromixmalFM: %f' % proximal_bayes.res['max']['max_val'])
    print(proximal_bayes.res['max']['max_params'])
















    # ################## TEST SET  #############################
    # X_test, y_test = reader.load_data("./xtest.csv")
    # Xn_test = reader.transform(X_test)
    # p_test = learner.predict(Xn_test)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # with open('../input/xsubmission_valid.csv', 'w') as outfile:
    #     outfile.write('id,Demanda_uni_equil\n')
    #     reader = DataFileReader('../input/xtest.ftrl', config['data_description'], config['model'])
    #     learner = Proximal(config=config['model'])
    #     print "Loading trainer weights n, z"
    #     learner.fromfile("../input/xtrain.pkl")
    #     for _, id, x, _ in reader.data():
    #         p, _ = learner.predict(x)
    #         outfile.write('%s,%.3f\n' % (id, expm1(max(0, p))))
    #         if((t % 1000000) == 0):
    #             print(t)
    # print('Finished')
