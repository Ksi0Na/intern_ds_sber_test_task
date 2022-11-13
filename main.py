import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def data_to_int(data):
    # delete unused column
    data.drop('Unnamed: 0', axis=1, inplace=True)

    # processing NaN values
    data.replace(np.nan, '0', inplace=True)

    # processing of the 'person_id' column
    data.iloc[:, 0] = data.iloc[:, 0].str.split('_').str[-1]

    # processing of the 'action_id' column
    data.iloc[:, 1] = data.iloc[:, 1].str.split('_')
    data.iloc[:, 1] = data.iloc[:, 1].str[0].str[-1] + data.iloc[:, 1].str[1]

    # processing of the 'date' column
    data.iloc[:, 2] = data.iloc[:, 2].str.split('-')
    data.iloc[:, 2] = data.iloc[:, 2].str[0] \
                      + data.iloc[:, 2].str[1] \
                      + data.iloc[:, 2].str[2]

    # processing of the 'action_type' and 'char_n' columns
    for i in range(3, len(data.iloc[0, :].index)):
        data.iloc[:, i] = data.iloc[:, i].str.split().str[-1]

    # converting types to int
    for i in range(0, len(data.iloc[0, :].index)):
        data.iloc[:, i] = pd.to_numeric(data.iloc[:, i])

    return data

# reading input files
train = pd.read_csv("../action_train.csv", dtype=str)
test = pd.read_csv("../action_test.csv", dtype=str)

# data transformation for training
train = data_to_int(train)
test = data_to_int(test)

# setting variables for training
x_train = train.drop('result', axis=1)
x_test = test
y_train = train.iloc[:, -1]

# model training
model = LogisticRegression()
model.fit(x_train, y_train)

# getting a prediction
predict = model.predict_proba(x_test)

# preparing data for output
output_data = {"action_id": test.iloc[:, 1],
               "result": predict[:, 1]}

# file creating and output
output_file = pd.DataFrame(output_data)
output_file.to_csv("../output.csv", sep=',', index=False)