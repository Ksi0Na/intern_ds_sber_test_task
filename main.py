import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def to_int_data(data):
    # delete unusable columns
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.drop('action_id', axis=1, inplace=True)
    data.drop('date', axis=1, inplace=True)

    # processing NaN values
    data.replace(np.nan, '0', inplace=True)

    # processing of the 'person_id' column
    data.iloc[:, 0] = data.iloc[:, 0].str.split('_').str[-1]

    # processing of the 'action_type' and 'char_n' columns
    for i in range(0, len(data.iloc[0, :].index)):
        data.iloc[:, i] = data.iloc[:, i].str.split().str[-1]

    # converting types to int
    for i in range(0, len(data.iloc[0, :].index)):
        data.iloc[:, i] = pd.to_numeric(data.iloc[:, i])

    return data

def to_int_person(data):
    # delete unusable columns
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.drop('date', axis=1, inplace=True)
    data.drop('group_1', axis=1, inplace=True)

    # processing bool values
    data.replace('False', '0', inplace=True)
    data.replace('True', '1', inplace=True)

    # renaming columns
    data.columns = data.columns.str.replace('har', '')

    # processing of the 'person_id' column
    data.iloc[:, 0] = data.iloc[:, 0].str.split('_').str[-1]

    # processing of the 'c_n' columns
    for i in range(0, len(data.iloc[0, :].index) - 1):
        data.iloc[:, i] = data.iloc[:, i].str.split().str[-1]

    # converting types to int
    for i in range(0, len(data.iloc[0, :].index)-1):
        data.iloc[:, i] = pd.to_numeric(data.iloc[:, i])

    return data

def take_concat_data(person,data):
    l = []
    for j in range(0, len(person.iloc[:, 0].index) - 1):
        for i in range(0, len(data.iloc[:, 0].index) - 1):
            if person.iloc[j, 0] == data.iloc[i, 0]:
                local = pd.concat([data.iloc[i, :], person.drop('person_id', axis=1).iloc[j, :]], axis=0)
                local.name = i
                l.append(local)

    return pd.DataFrame(l)

# reading input files
train = pd.read_csv("../action_train.csv", dtype=str)
test = pd.read_csv("../action_test.csv", dtype=str)
person = pd.read_csv("../person.csv", dtype=str)

# saving and cleaning some data
y_train = train.iloc[:, -1].astype(int)
train.drop('result', axis=1, inplace=True)
action_id = test['action_id']

# data transformation for training
train = to_int_data(train)
test = to_int_data(test)
person = to_int_person(person)
train = take_concat_data(person, train)
test = take_concat_data(person, test)

# setting variables for training
x_train = train.drop('result', axis=1)
x_test = test

# model training
model = LogisticRegression()
model.fit(x_train, y_train)

# getting a prediction
predict = model.predict_proba(x_test)

# preparing data for output
output_data = {"action_id": action_id,
               "result": predict[:, 1]}

# file creating and output
output_file = pd.DataFrame(output_data)
output_file.to_csv("../output.csv", sep=',', index=False)