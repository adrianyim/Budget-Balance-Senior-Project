from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.contrib.auth import get_user_model
from django.db.models import Sum, Q, F

import datetime

from sqlalchemy import create_engine

from .models import Item
from . import connectpsql

import pandas as pd
import numpy as np
from numpy import exp, array, random, dot
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import models
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima.arima.utils import ndiffs
from TFANN import ANNR

from pandas.plotting import autocorrelation_plot

import matplotlib.pyplot as plt

from flask import Flask, render_template

# Point to CustomUser table
User = get_user_model()

# Create your views here.
class Perceptron(object):
    # Implements a perceptron network
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

class NeuronNetwork(object):
    def __init__(self):
        random.seed(1)
        
        self.synaptic_weights = 2 * random.random() - 1

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def learning(self, inputs):
        return self._sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, inputs, outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.learning(inputs)

            error = outputs - output

            adjustment = dot(inputs, error * self._sigmoid_derivative(output))

            self.synaptic_weights += adjustment

def processPrediction(dfDay, history, prediction_days):
    last_date = dfDay.iloc[[-1]].index + datetime.timedelta(days=1)

    # orginal days list
    dfDays = pd.DataFrame(columns=["date", "cost"])
    dfDays.date = dfDay.index
    dfDays.cost = dfDay.tolist()
    dfDays.set_index("date", inplace=True)

    # predict days list
    dfPredict = pd.DataFrame(columns=["date", "cost"])
    dfPredict.date = pd.date_range(last_date[0], periods=prediction_days, freq="D")
    dfPredict.cost = history[-prediction_days:]
    dfPredict.set_index("date", inplace=True)

    # Combine two data lists
    dfDays = dfDays.append(dfPredict)

    # plt.plot(dfDays.index, dfDays)
    # plt.show( )
    
    return dfDays, dfPredict

def inverse_diffference(history, predict, interval=1):
    return predict + history[-interval]

def difference(df, interval=1):
    diff_list = []

    for i in range(interval, len(df)):
        value = df[i] - df[i - interval]
        diff_list.append(value)

    return array(diff_list)

def convertToNum(df):
    df = df.replace(to_replace="Income", value=0)
    df = df.replace(to_replace="Expense", value=1)
    df = df.replace(to_replace="Salaries and wages", value=10)
    df = df.replace(to_replace="Utility expenses", value=11)
    df = df.replace(to_replace="Administration expenses", value=12)
    df = df.replace(to_replace="Finance costs", value=13)
    df = df.replace(to_replace="Depreciation", value=14)
    df = df.replace(to_replace="Impairment losses", value=15)
    df = df.replace(to_replace="Food", value=16)
    df = df.replace(to_replace="Others", value=17)

    return df

def processingDataset(dataset):
    pd.set_option('display.max_rows', None)

    # Create a DataFrame
    df = pd.DataFrame(columns=["date", "itemType", "costType", "cost"])
    df.date = dataset.date.tolist()
    df.itemType = dataset.item_type.tolist()
    df.costType = dataset.cost_type.tolist()
    df.cost = dataset.cost.tolist()

    # Set date to be df.index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    # Convert string to num
    df = convertToNum(df)

    # Filter costType
    expensesList = df[df.costType==1]
    expensesList = expensesList.drop(columns="costType")

    # Sort by date, item type, and cost type
    dfGroup = expensesList.groupby(["date", "itemType"]).sum()
    # Daily total
    dfDay = expensesList.cost.resample('D').sum()
    # Weekly total
    dfWeek = expensesList.cost.resample('W').sum()
    # Monthly total
    dfMonth = expensesList.cost.resample('M').sum()

    # dfWeekDay = pd.merge(dfWeek, dfDay, how="outer", on="date", sort=True, suffixes=("_week", "_day"))
    # dfMonth = pd.merge(dfMonth, dfWeekDay, how="outer", on="date", sort=True)
    # dfMonth = dfMonth.replace(to_replace=np.nan, value=-1)

    return expensesList, round(dfDay, 2)

def predict():
    # Connect to psql server
    engine = create_engine(connectpsql.psql)
    sql_command = "SELECT date, item_type, cost_type, cost FROM budget_item ORDER BY date"

    # Read dataset from psql server
    dataset = pd.read_sql(sql_command, engine, parse_dates=["date"])
    
    expensesList, dfDay = processingDataset(dataset)

    # autocorrelation_plot(expensesList)
    # plt.show()

    # plt.scatter(expensesList.index, expensesList.cost)
    # plt.scatter(df.date, df.cost, c=0.5)

    # plt.plot(dfDay.index, dfDay, "b-")
    # plt.xlabel("Date")
    # plt.ylabel("cost", color="b")
    # plt.show()

    # ##------------------------------------------------------------------##
    ## Plot residual errors
    # residuls = pd.DataFrame(model_fit.resid)
    # residuls.plot(kind="kde")
    # plt.show()
    # print(residuls.describe())

    # totalweek = len(dfDay)
    # lastweek = 1
    # pastweek = totalweek - lastweek
    # train, test = dfDay[0:pastweek], dfDay[pastweek:totalweek]
    # # train, test= train_test_split(dfDay, test_size = 0.1)

    # history = [x for x in train]
    # predictions_list = []

    # 5, 1, 0 > 188.426 497.815 (630.170)
    # 5, 0, 1 > 1.549   445.254 564.480
    # 6, 0, 1 > 1.671   438.906 561.012
    # 7, 0, 1 > 1.819   448.877

    # for i in range(len(test)):
    #     model = ARIMA(history, order=(6, 0, 1))
    #     model_fit = model.fit(disp=0)
    #     target = model_fit.forecast()
    #     predictions_list.append(target[0])
    #     history.append(test[i])
    #     print("predicted=%f, actual=%f" % (target[0], test[i]))

    # error = mean_squared_error(test, predictions_list)
    # print("Test MSE: %.3f" % error)

    # plt.plot(test, label="Actuals")
    # plt.plot(predictions_list, label="forecast", color="red")
    # plt.show()
    
    days_in_month = 31      # Comparing past days
    prediction_days = 7    # Predicting days

    diff_list = difference(dfDay, days_in_month)
    model = ARIMA(diff_list, order=(5, 0, 1))
    model_fit = model.fit(disp=0)
    # start_index = len(diff_list)
    # end_index = start_index + 29
    # forecast = model_fit.predict(start=start_index, end=end_index)
    forecast = model_fit.forecast(steps=prediction_days)[0]
    # forecast = inverse_diffference(dfDay, forecast, days_in_month)
    # print("Forecast: %f" % forecast)
    history = [x for x in dfDay]

    for predict in forecast:
        inverted = inverse_diffference(history, predict, days_in_month)
        if inverted < 0:
            inverted = 0.0
        history.append(round(inverted, 2))

    dfDays, dfPredict = processPrediction(dfDay, history, prediction_days)

    predict_html = dfPredict.to_html(header=False, index_names=False, border=0, classes="predictTable")
    # return render_template("prediction.html", tables=[dfPredict.to_html(classes="predic_table")])

    return predict_html

def deleteItems(request, id):
    Item.objects.filter(id = id).delete()
    return redirect("home")

def updateItems(request, id):
    if request.POST:
        item = request.POST["item"]
        item_type = request.POST["item_type"]
        cost = request.POST["cost"]
        cost_type = request.POST["cost_type"]
        remark = request.POST["remark"]
        date = request.POST["date"]

        Item.objects.filter(id = id).update(
            item = item, 
            item_type = item_type, 
            cost = cost, 
            cost_type = cost_type, 
            remark = remark,
            date = date)

        return redirect("home")
    else:
        item = get_object_or_404(Item, id=id)

        return render(request, "updateItems.html", {
            "item":item
        })

def insertItems(request):
    item = request.POST["item"]
    item_type = request.POST["item_type"]
    cost = request.POST["cost"]
    cost_type = request.POST["cost_type"]
    remark = request.POST["remark"]

    itemDB = Item(
        item = item, 
        item_type = item_type, 
        cost = cost, 
        cost_type = cost_type, 
        remark = remark,
        username = User.objects.get(username = request.user))
    itemDB.save()

    return redirect("home")

def home(request):
    if request.user.is_authenticated:
        items = Item.objects.filter(username = request.user).order_by("-date") # order by date
        expense = calculateExpenseToTal(request)
        income = calculateIncomeToTal(request)
        predict_list = predict()

        return render(request, "home.html", {
            "items": items, 
            "expense": expense,
            "income": income,
            "predict_list": predict_list
            })
    else:
        return redirect("/user/login")

def calculateExpenseToTal(request):
    username = Q(username=request.user)
    costtype = Q(cost_type="Expense")
    expense = Item.objects.filter(username, costtype).aggregate(total_sum=Sum("cost"))["total_sum"]

    return expense

def calculateIncomeToTal(request):
    username = Q(username=request.user)
    costtype = Q(cost_type="Income")
    income = Item.objects.filter(username, costtype).aggregate(total_sum=Sum("cost"))["total_sum"]

    return income

# pg_ctl.exe start -D C:\Users\adrian\Apps\PostgreSQL\data
