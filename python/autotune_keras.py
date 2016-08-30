__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
import datetime
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dropout, Activation, Dense, regularizers
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score as auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import logging


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = auc(self.y_val, y_pred)
            logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))


def kerascv(dense1, dense2, epochs):
    ival = IntervalEvaluation(validation_data=(x1, y1), interval=1)

    pred_sum = 0
    for k in range(1):
        model = Sequential()
        model.add(Dense(int(dense1), input_shape=(dims,), init='he_uniform', W_regularizer=regularizers.l1(0.0005)))
        model.add(Dropout(0.05))#    input dropout
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(int(dense2)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer="adam")
        model.fit(x0, y0, nb_epoch=int(epochs), batch_size=128, verbose=0, callbacks=[ival])


        preds = model.predict_proba(x1, batch_size=64, verbose=0)[:,1]
        pred_sum += preds
        pred_average = pred_sum / (k+1)
        del model

    loss = auc(y1[:,1],pred_average)
    return loss


def getDummy(df,col):
        category_values=df[col].unique()
        data=[[0 for i in range(len(category_values))] for i in range(len(df))]
        dic_category=dict()
        for i,val in enumerate(list(category_values)):
            dic_category[str(val)]=i
        for i in range(len(df)):
            data[i][dic_category[str(df[col][i])]]=1

        data=np.array(data)
        for i,val in enumerate(list(category_values)):
            df.loc[:,"_".join([col,str(val)])]=data[:,i]

        return df

if __name__ == "__main__":

    ## settings
    projPath = './'
    dataset_version = "v1"
    model_type = "keras"
    seed_value = 789775
    todate = datetime.datetime.now().strftime("%Y%m%d")

    ## data
    xtrain = pd.read_csv(projPath + 'input/xvalid_20160821.csv')
    id_train = xtrain.activity_id
    y_train = xtrain.outcome
    xtrain.drop('activity_id', axis = 1, inplace = True)
    xtrain.drop('outcome', axis = 1, inplace = True)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train).astype(np.int32)
    y_train = np_utils.to_categorical(y_train)

    print ("processsing finished")
    xtrain = np.array(xtrain)
    xtrain = xtrain.astype(np.float32)

    scaler = StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)

    # folds
    xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
    # work with 5-fold split
    fold_index = np.array(xfolds.fold5)

    # work with validation split
    idx0 = np.where(fold_index != 0)
    idx1 = np.where(fold_index == 0)
    x0 = xtrain[idx0,:][0]
    x1 = xtrain[idx1,:][0]
    y0 = y_train[idx0]
    y1 = y_train[idx1]

    nb_classes = 2
    dims = xtrain.shape[1]
    print(dims, 'dims')

    kerasBO = BayesianOptimization(kerascv,
                                   {#'dropout_init':(0.01, 0.1),
                                    'dense1': (int(0.15 * xtrain.shape[1]), int(5 * xtrain.shape[1])),
                                    #'dropout1': (0.15, 0.5),
                                    'dense2': (int(0.15 * xtrain.shape[1]), int(5 * xtrain.shape[1])),
                                    #'dropout2': (0.05, 0.5),
                                    'epochs': (int(20), int(75))
                                    })

    kerasBO.maximize(init_points=3, n_iter=25)
    print('-' * 53)

    print('Final Results')
    print('Keras: %f' % kerasBO.res['max']['max_val'])
    print(kerasBO.res['max']['max_params'])
