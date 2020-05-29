import math
import pandas as pd
import numpy as np
import datetime
import connectpsql
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

pd.set_option('display.max_rows', None)

# fill out the missing data in front of the dataset
def add_first_day():
    date_order = pd.date_range('2019-09-01', '2019-09-16')
    date_range = pd.DataFrame(data=0, index=date_order, columns=["cost"])
    return date_range

# fill out the missing data in back of the dataset
def add_last_day(df):
    today = datetime.datetime.today()
    last_date = df.index[-1] + datetime.timedelta(days=1)
    date_order = pd.date_range(last_date, today)
    date_range = pd.DataFrame(data=0, index=date_order, columns=["cost"])
    return date_range

# process the datset to be ready to put into the ML algorithm 
def process_dataset(dataset, flag):
    # create dataframe
    df = pd.DataFrame(dataset).copy()
    df.set_index("date", inplace=True)

    # sum the daily total
    df = df.resample('D').sum()

    # income set
    if flag == 0:
        first_date_range = add_first_day()
        last_date_range = add_last_day(df)
        
        df = pd.concat([first_date_range, df, last_date_range])

    # expense set
    if flag == 1:
        last_date_range = add_last_day(df)
        df = pd.concat([df, last_date_range])

    # df = df.replace(to_replace=np.nan, value=0)

    return df

# convert all the strings to numbers
def convert_to_num(df):
    df = df.replace(to_replace="Income", value=0)
    df = df.replace(to_replace="Expense", value=1)
    df = df.replace(to_replace="Salaries and wages", value=100)
    df = df.replace(to_replace="Utility expenses", value=110)
    df = df.replace(to_replace="Administration expenses", value=120)
    df = df.replace(to_replace="Finance costs", value=130)
    df = df.replace(to_replace="Depreciation", value=140)
    df = df.replace(to_replace="Impairment losses", value=150)
    df = df.replace(to_replace="Food", value=160)
    df = df.replace(to_replace="Others", value=170)

    return df

# connect psql server to get the dataset
engine = create_engine(connectpsql.psql)
sql_command = 'SELECT date, item_type, cost_type, cost FROM budget_new_item ORDER BY date'
dataset = pd.read_sql(sql_command, engine, parse_dates=['date'])

dataset = convert_to_num(dataset)

# income set
df_income = dataset[dataset.cost_type==0].drop(['cost_type', 'item_type'], axis=1)
df_income = process_dataset(df_income, 0)

# expense set
df_expense = dataset[dataset.cost_type==1].drop(['cost_type', 'item_type'], axis=1)
df_expense = process_dataset(df_expense, 1)

# scale the data
scaler = MinMaxScaler()
scaled_income = scaler.fit_transform(df_income)
scaled_expense = scaler.fit_transform(df_expense)

# training length of data
income_training_len = math.ceil(len(scaled_income) * 0.8)
expense_training_len = math.ceil(len(scaled_expense) * 0.8)

# create income training set
income_training_set = scaled_income[0:income_training_len]
expense_training_set = scaled_expense[0:expense_training_len]

# data of the past days
DAY = 30

x_train_income = []
y_train_income = []
x_train_expense = []
y_train_expense = []

# split data into x_train_, y_train_ data sets
for i in range(DAY, len(income_training_set)):
    x_train_income.append(income_training_set[i-DAY:i])
    y_train_income.append(income_training_set[i,0])

for i in range(DAY, len(expense_training_set)):
    x_train_expense.append(expense_training_set[i-DAY:i])
    y_train_expense.append(expense_training_set[i,0])

# convert the training sets to numpy arrays
x_train_income, y_train_income = np.array(x_train_income), np.array(y_train_income)
x_train_expense, y_train_expense = np.array(x_train_expense), np.array(y_train_expense)

# build a LSTM model
model = Sequential()

model.add(LSTM(30, return_sequences=True, input_shape=(x_train_income.shape[1], x_train_income.shape[2])))
model.add(LSTM(60, return_sequences=True))
model.add(LSTM(90, return_sequences=True))
model.add(LSTM(60, return_sequences=True))
model.add(LSTM(30))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# train the model
# model.fit(x_train_income, y_train_income, batch_size=1, epochs=5)
model.fit(x_train_expense, y_train_expense, batch_size=1, epochs=5)

# testing data set
income_testing_set = scaled_income[income_training_len - DAY:]
expense_testing_set = scaled_expense[expense_training_len - DAY:]

x_test_income = []
y_test_income = df_income[income_training_len:]
x_test_expense = []
y_test_expense = df_expense[expense_training_len:]

# split data into x_test, y_test data sets
for i in range(DAY, len(income_testing_set)):
    x_test_income.append(income_testing_set[i-DAY:i, 0])

for i in range(DAY, len(expense_testing_set)):
    x_test_expense.append(expense_testing_set[i-DAY:i, 0])

# convert the testing set to a numpy array
x_test_income = np.array(x_test_income)
x_test_expense = np.array(x_test_expense)

# reshape the testing set
x_test_income = np.reshape(x_test_income, (x_test_income.shape[0], x_test_income.shape[1], 1))
x_test_expense = np.reshape(x_test_expense, (x_test_expense.shape[0], x_test_expense.shape[1], 1))

# predict the values
income_predictions = model.predict(x_test_income)
income_predictions = scaler.inverse_transform(income_predictions)

expense_predictions = model.predict(x_test_expense)
expense_predictions = scaler.inverse_transform(expense_predictions)

# root mean square error (RMSE)
rmse_income = np.sqrt(np.mean(income_predictions - y_test_income)**2)
rmse_expense = np.sqrt(np.mean(expense_predictions - y_test_expense)**2)
print("Expense error: {0}".format(rmse_expense))
print("Income error: {0}".format(rmse_income))

# plot the data
train_income = df_income[:income_training_len]
valid_income = df_income[income_training_len:]
valid_income['predictions'] = income_predictions

train_expense = df_expense[:expense_training_len]
valid_expense = df_expense[expense_training_len:]
valid_expense['predictions'] = expense_predictions
print(valid_income)
print(valid_expense)

# visualize the data
# plt.figure(figsize=(10,6))
# plt.title('LSTM model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Income USD ($)', fontsize=18)
# plt.plot(train_income['cost'])
# plt.plot(valid_income[['cost', 'predictions']])
# plt.legend(['Train', 'Valid', 'Predictions'], loc='best')
# plt.show()