# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class BinaryStackingClassifierNP():
    """
    To facilitate stacking of binary classifiers for numpy specific arrays or sparse arrays
     - Really this is to allow ProximalFM stacking.
    It only provides fit and predict_proba functions, and works with binary [0, 1] labels.
    :param base_classifiers: A list of binary classifiers with a fit and predict_proba method similar to that of sklearn
    :param xfolds: A cross folds pandas data frame indicating the identifier for fold selection (col1) and the fold
                   number (col2)
                   ID,fold5
                   3,   3
                   4,   5
                   5,   3
                   6,   3
                   8,   1
                   In this class the names do not matter it is positional.
                   ##############################
                   Fold number must start from 0. as per Kfold or LabelKfold in scikit-learn.
                   ##############################
    :param evaluation: optional evaluation metric (y_true, y_score) to check metric at each fold.
                    expected use case might be evaluation=sklearn.Metrics.logLoss
    """
    def __init__(self, base_classifiers, xfolds, evaluation=None):
        self.base_classifiers = base_classifiers
        assert(xfolds.shape[1] == 2)
        self.xfolds = xfolds
        self.evaluation = evaluation

        # Build an empty pandas dataframe to store the meta results to.
        # As many rows as the folds data, as many cols as base classifiers
        self.colnames = ["v" + str(n) for n in range(len(self.base_classifiers))]
        # Check we have as many colnames as classifiers
        self.stacking_train = pd.DataFrame(np.nan, index=self.xfolds.index, columns=self.colnames)

    def fit(self, X, y, **kwargs):
        """ A generic fit method for meta stacking.
        :param X: A train dataset
        :param y: A train labels
        :param kwargs: Any optional params to give the fit method, i.e in xgboost we may use eval_metirc='auc'
        :return:
        """
        # Loop over the different classifiers.
        fold_index = self.xfolds.ix[:, 1]
        fold_index = np.array(fold_index)
        n_folds = len(np.unique(fold_index))

        for model_no in range(len(self.base_classifiers)):
            print "Running Model ", model_no+1, "of", len(self.base_classifiers)
            loss_avg = 0
            for j in range(n_folds):
                idx0 = np.where(fold_index != j)
                idx1 = np.where(fold_index == j)
                idx1pd = self.xfolds[self.xfolds.ix[:,1] == j].index
                x0 = X[idx0]
                x1 = X[idx1]
                y0 = y[idx0]
                y1 = y[idx1]
                self.base_classifiers[model_no].fit(x0, y0, **kwargs)
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(x1)
                if self.evaluation is not None:
                    loss = self.evaluation(y1, predicted_y_proba)
                    print "Current Fold Loss = ", loss
                    loss_avg += loss
                self.stacking_train.ix[self.stacking_train.index.isin(idx1pd), model_no] = predicted_y_proba.ravel()
            print "Model CV-Loss across folds =", loss_avg / n_folds
            # Finally fit against all the data
            self.base_classifiers[model_no].fit(X, y, **kwargs)

    def predict(self, X):
        """
        :param X: The data to apply the fitted model from fit
        :return: The predict of the different classifiers - so other methods can be used i.e regressors or clusters
        """
        stacking_predict_data = pd.DataFrame(np.nan, index=np.arange(X.shape[0]), columns=self.colnames)

        for model_no in range(len(self.base_classifiers)):
            stacking_predict_data.ix[:, model_no] = self.base_classifiers[model_no].predict(X)
        return stacking_predict_data

    def predict_proba(self, X):
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted_proba of the different classifiers
        """
        stacking_predict_proba_data = pd.DataFrame(np.nan, index = np.arange(X.shape[0]), columns=self.colnames)

        for model_no in range(len(self.base_classifiers)):
            stacking_predict_proba_data.ix[:, model_no] = self.base_classifiers[model_no].predict_proba(X).ravel()
        return stacking_predict_proba_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict_proba method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict_proba[:, 1]
        """
        return self.stacking_train.copy()