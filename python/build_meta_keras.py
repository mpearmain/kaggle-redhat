import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense, regularizers
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
import datetime

class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = auc(self.y_val, y_pred)
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

def createModel():
    model = Sequential()
    model.add(Dense(int(85), input_shape=(dims,), init='he_uniform', W_regularizer=regularizers.l1(0.0005)))
    model.add(Dropout(0.05))  # input dropout
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(int(83)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam")
    return model


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
xtrain.drop('activity_id', axis=1, inplace=True)
xtrain.drop('outcome', axis=1, inplace=True)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int32)
y_train = np_utils.to_categorical(y_train)

xtest = pd.read_csv(projPath + 'input/xtest_ds_' + dataset_version + '.csv')
id_test = xtest.activity_id
xtest.drop('activity_id', axis=1, inplace=True)


print ("processsing finished")
xtrain = np.array(xtrain)
xtrain = xtrain.astype(np.float32)
xtest = np.array(xtest)
xtest = xtest.astype(np.float32)

# folds
xfolds = pd.read_csv(projPath + 'input/5-fold.csv')
# work with 5-fold split
fold_index = np.array(xfolds.fold5)
# work with 5-fold split
fold_index = xfolds.fold5
fold_index = np.array(fold_index) - 1
n_folds = len(np.unique(fold_index))

nb_classes = 2
print nb_classes, 'classes'

nb_classes = 2
dims = xtrain.shape[1]
print(dims, 'dims')

auc_scores=[]
best_score=-1

# storage structure for forecasts
mvalid = np.zeros((xtrain.shape[0],1))
mfull = np.zeros((xtest.shape[0],1))

nbag =1

## build 2nd level forecasts
for i in range(1):
        print "Combo:", i+1, "of", 1
        # loop over folds
        # Recompile model on each fold
        for j in range(0,n_folds):
            # configure model with j-th combo of parameters
            idx0 = np.where(fold_index != j)
            idx1 = np.where(fold_index == j)
            x0 = np.array(xtrain)[idx0,:][0]
            x1 = np.array(xtrain)[idx1,:][0]
            y0 = np.array(y_train)[idx0]
            y1 = np.array(y_train)[idx1]

            # setup bagging classifier
            pred_sum = 0
            for k in range(nbag):
                ival = IntervalEvaluation(validation_data=(x1, y1), interval=5)
                model = createModel()
                print 'Fold:', j, 'Building bag:', k
                model.fit(x0, y0, nb_epoch=3, batch_size=128, verbose=0, callbacks=[ival])
                preds = model.predict_proba(x1, verbose=0)[:,1]
                pred_sum += preds
                pred_average = pred_sum / (k+1)
                print 'AUC val:', auc(y1[:,1], pred_average)
                del model
                print 'Finished bag:', k
            mvalid[idx1,i] = pred_average
            print "finished fold:", j

        print "Building full prediction model for test set."
        # configure model with j-th combo of parameters

        # setup bagging classifier
        pred_sum = 0
        for k in range(nbag):
            model = createModel()
            print 'Building bag:', k
            model.fit(xtrain, y_train, nb_epoch=43, batch_size=128, verbose=0)
            preds = model.predict_proba(np.array(xtest), verbose=0)[:,1]
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
mvalid['activity_id'] = id_train
mvalid['outcome'] = pred_average

mfull = pd.DataFrame(mfull)
mfull.columns = [model_type + str(i) for i in range(0, mfull.shape[1])]
mfull['activity_id'] = id_test


# save the files
mvalid.to_csv('./metafeatures/prval2_' + model_type + '_' + todate + '_data' + model_type + '_seed' + str(seed_value) + '.csv', index = False, header = True)
mfull.to_csv('./metafeatures/prfull2_' + model_type + '_' + todate + '_data' + model_type + '_seed' + str(seed_value) + '.csv', index = False, header = True)
