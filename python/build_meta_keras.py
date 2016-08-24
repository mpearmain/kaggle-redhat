import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense, regularizers
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import datetime

## settings
dataset_version = "kb10x06"
nbag = 10
model_type = 'keras_2layer_bag' + str(nbag)
seed_value = 260681
todate = datetime.datetime.now().strftime("%Y%m%d")
np.random.seed(seed_value)
need_normalise=True


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print "interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score)


def getDummy(df,col):
    category_values=df[col].unique()
    data=[[0 for i in range(len(category_values))] for i in range(len(df))]
    dic_category=dict()
    for i,val in enumerate(list(category_values)):
        dic_category[str(val)]=i
   # print dic_category
    for i in range(len(df)):
        data[i][dic_category[str(df[col][i])]]=1

    data=np.array(data)
    for i,val in enumerate(list(category_values)):
        df.loc[:,"_".join([col,str(val)])]=data[:,i]

    return df

def createModel(x):
    model = Sequential()
    model.add(Dense(x[0], input_shape=(dims,), init='he_uniform', W_regularizer=regularizers.l1(0.0001)))
    model.add(Activation('relu'))
    model.add(Dropout(x[1]))# input dropout
    model.add(Dense(x[2], init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(x[3]))
    model.add(Dense(nb_classes, init='he_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adagrad")
    return model


train = pd.read_csv('../input/xtrain_'+ dataset_version + '.csv')
id_train = train.ID
y_train_target = train.TARGET
y_train = train.TARGET
train.drop('ID', axis = 1, inplace = True)
train.drop('TARGET', axis = 1, inplace = True)

test = pd.read_csv('../input/xtest_'+ dataset_version + '.csv')
id_test = test.ID
test.drop('ID', axis = 1, inplace = True)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int32)
y_train = np_utils.to_categorical(y_train)

print ("processsing finished")
train = np.array(train)
train = train.astype(np.float32)
test = np.array(test)
test = test.astype(np.float32)
if need_normalise:
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

# folds
xfolds = pd.read_csv('../input/xfolds.csv')
# work with 5-fold split
fold_index = xfolds.fold5
fold_index = np.array(fold_index) - 1
n_folds = len(np.unique(fold_index))

nb_classes = 2
print nb_classes, 'classes'

dims = train.shape[1]
print dims, 'dims'

auc_scores=[]
best_score=-1

param_grid = [[350, 0.05, 2788, 0.5, 125, 256]]
# storage structure for forecasts
mvalid = np.zeros((train.shape[0],len(param_grid)))
mfull = np.zeros((test.shape[0],len(param_grid)))

## build 2nd level forecasts
for i in range(len(param_grid)):
        print "processing parameter combo:", param_grid[i]
        print "Combo:", i+1, "of", len(param_grid)
        # loop over folds
        # Recompile model on each fold
        for j in range(0,n_folds):
            # configure model with j-th combo of parameters
            x = param_grid[i]
            idx0 = np.where(fold_index != j)
            idx1 = np.where(fold_index == j)
            x0 = np.array(train)[idx0,:][0]
            x1 = np.array(train)[idx1,:][0]
            y0 = np.array(y_train)[idx0]
            y1 = np.array(y_train)[idx1]

            # setup bagging classifier
            pred_sum = 0
            for k in range(nbag):
                ival = IntervalEvaluation(validation_data=(x1, y1), interval=10)
                model = createModel(x)
                print 'Fold:', j, 'Building bag:', k
                model.fit(x0, y0, nb_epoch=x[4], batch_size=x[5], verbose=0, callbacks=[ival])
                preds = model.predict_proba(x1, verbose=0)[:,1]
                pred_sum += preds
                pred_average = pred_sum / (k+1)
                print 'AUC val:', roc_auc_score(y1, pred_average)
                del model
                print 'Finished bag:', k
            mvalid[idx1,i] = pred_average
            print "finished fold:", j

        print "Building full prediction model for test set."
        # configure model with j-th combo of parameters
        x = param_grid[i]

        # setup bagging classifier
        pred_sum = 0
        for k in range(nbag):
            model = createModel(x)
            print 'Building bag:', k
            model.fit(np.array(train), y_train, nb_epoch=x[4], batch_size=x[5], verbose=0)
            preds = model.predict_proba(np.array(test), verbose=0)[:,1]
            pred_sum += preds
            pred_average = pred_sum / (k+1)
            del model
            print 'Finished bag:', k
        mfull[:,i] = pred_average
        print "finished full prediction"

## store the results
# add indices etc
mvalid = pd.DataFrame(mvalid)
mvalid.columns = [model_type + str(i) for i in range(0, mvalid.shape[1])]
mvalid['ID'] = id_train
mvalid['TARGET'] = y_train_target

mfull = pd.DataFrame(mfull)
mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
mfull['ID'] = id_test


# save the files
mvalid.to_csv('../metafeatures/prval_' + model_type + '_' + todate + '_data' + model_type + '_seed' + str(seed_value) + '.csv', index = False, header = True)
mfull.to_csv('../metafeatures/prfull_' + model_type + '_' + todate + '_data' + model_type + '_seed' + str(seed_value) + '.csv', index = False, header = True)
