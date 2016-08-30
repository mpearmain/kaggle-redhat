__author__ = 'michael.pearmain'

import datetime
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc
from bayes_opt import BayesianOptimization


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              subsample,
              colsample_bytree,
              gamma,
              min_child_weight,
              silent=True,
              seed=1234):


    params = {
        "objective": "binary:logistic",
        "booster": "gblinear",
        "eval_metric": "auc",
        "eta": learning_rate,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": silent,
        "seed": seed,
    }
    num_boost_round = int(n_estimators)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params,
                    dtrain,
                    num_boost_round,
                    evals=watchlist,
                    verbose_eval=False)

    print("Validating...")
    check = gbm.predict(dvalid)
    score = auc(dvalid.get_label(), check)

    return score

if __name__ == "__main__":

    projPath = './'
    model_type = "xgb_hash"
    seed_value = 674839
    todate = datetime.datetime.now().strftime("%Y%m%d")

    dtrain = xgb.DMatrix(projPath + 'input/dmodel.data')
    dvalid = xgb.DMatrix(projPath + 'input/dvalid.data')

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (int(6), int(20)),
                                      'learning_rate': (0.01, 0.4),
                                      'n_estimators': (int(500), int(2500)),
                                      'subsample': (0.6, 0.9),
                                      'colsample_bytree': (0.6, 0.9),
                                      'gamma': (0.0, 0.1),
                                      'min_child_weight': (int(0), int(3))
                                     })

    xgboostBO.maximize(init_points=5, n_iter=50, xi=0.05)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
    print('XGBOOST: %s' % xgboostBO.res['max']['max_params'])
