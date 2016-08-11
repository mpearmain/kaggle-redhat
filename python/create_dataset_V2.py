from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import random


random.seed(2016)

def preprocess_acts(data, trainset = True):
    data = data.drop(['activity_id'], axis=1)

    columns = list(data.columns)
    columns.remove('date')
    if trainset:
        columns.remove('outcome')

    print "Processing dates"
    data['tyear'] = data['date'].dt.year
    data['tmonth'] = data['date'].dt.month
    data['tyearweek'] = data['date'].dt.week
    data['tday'] = data['date'].dt.day

    ## Split off from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    # Convert strings to ints
    for col in columns[1:]:
        print "Processing", col
        data[col] = data[col].fillna('type 0')
        le = LabelEncoder()
        data[col] = data.groupby(le.fit_transform(data[col]))[col].transform('count') / data.shape[0]

    data['t_sum_true'] = data[columns[2:11]].sum(axis=1)

    return data


def preprocess_people(data):
    ## Split off from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    #  Values in the people df is Booleans and Strings
    columns = list(data.columns)
    bools = columns[12:-1]
    strings = columns[1:12]
    strings.remove('date')

    for col in bools:
        print "Processing", col
        data[col] = pd.to_numeric(data[col]).astype(int)
    # Get the sum of positive results - not including 38
    data['p_sum_true'] = data[bools[:-1]].sum(axis=1)

    # Rather than turning them into ints which is fine for trees, lets develop response rates
    # So they can be used in other models.
    for col in strings:
        data[col] = data[col].fillna('type 0')
        le = LabelEncoder()
        data[col] = data.groupby(le.fit_transform(data[col]))[col].transform('count') / data.shape[0]

    print "Processing dates"
    data['pyear'] = data['date'].dt.year
    data['pmonth'] = data['date'].dt.month
    data['pyearweek'] = data['date'].dt.week
    data['pday'] = data['date'].dt.day

    print "People processed"
    return data

def read_test_train():
    ####### Build and save the datsets
    print("Read people.csv...")
    people = pd.read_csv("./input/people.csv",
                         dtype={'people_id': np.str,
                                'activity_id': np.str,
                                'char_38': np.int32},
                         parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("./input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])
    id_train = train['activity_id']

    print("Load test.csv...")
    test = pd.read_csv("./input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])
    id_test = train['activity_id']
    # Preprocess each df
    peeps = preprocess_people(people)
    actions_train = preprocess_acts(train)
    actions_test = preprocess_acts(test, trainset=False)

    # Training
    print "Merge Train set"
    train = actions_train.merge(peeps, how='left', on='people_id')
    train['activity_id'] = id_train
    print "Merge Test set"
    # Testing
    test = actions_test.merge(peeps, how='left', on='people_id')
    test['activity_id'] = id_test

    # Play with dates:
    train['days_diff'] = [int(i.days) for i in (train.date_x - train.date_y)]
    test['days_diff'] = [int(i.days) for i in (test.date_x - test.date_y)]

    # finally remove date vars:
    train.drop(['date_x', 'date_y'], axis=1, inplace=True)
    test.drop(['date_x', 'date_y'], axis=1, inplace=True)

    return train, test


train, test= read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))

train.to_csv("./input/xtrain_ds_v2.csv", index=False, header=True)
test.to_csv("./input/xtest_ds_v2.csv", index=False, header=True)

print "Done"