from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.models import User, auth
from django.contrib.auth import get_user_model
from django.db.models import Sum, Q, F
from django.contrib import messages
from django.views.generic import FormView
from rest_framework.views import APIView
from rest_framework.response import Response 
from sqlalchemy import create_engine
from .mixins import AjaxFormMixin
from .forms import ItemForm, DayForm
from .models import Item
from . import connectpsql
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy import exp, array, random, dot
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
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
from flask import Flask, render_template
from django.views.decorators.csrf import csrf_exempt, csrf_protect
# app = Flask(__name__)

# point to CustomUser table
User = get_user_model()

# fill up the empty rows with zero 
def insertZero(costList):
    # daily total
    dfDay = costList.cost.resample('D').sum()

    today = datetime.datetime.today() #yyyy-mm-dd
    last_date = dfDay.iloc[[-1]].index # find the last date of dfDay
    
    # add zero until today
    while last_date < today - datetime.timedelta(days=1):
        last_date += datetime.timedelta(days=1) # add 1 day
        new_row = pd.Series(data={" ": 0}, index=last_date) # create a new row
        dfDay = dfDay.append(new_row, ignore_index=False) # insert into dfDay

    dfDay = dfDay.replace(to_replace=np.nan, value=0)

    return round(dfDay, 2)

# predicting
def processPrediction(dfDay, history, prediction_days):
    last_date = dfDay.iloc[[-1]].index + datetime.timedelta(days=1)

    ## orginal days list
    dfOrginal = pd.DataFrame(columns=["date", "cost"])
    # dfOrginal = pd.DataFrame(columns=["cost"])
    dfOrginal.date = dfDay.index
    dfOrginal.cost = dfDay.tolist()
    # dfOrginal.set_index("date", inplace=True)

    ## predict days list
    dfPredict = pd.DataFrame(columns=["date", "cost"])
    dfPredict.date = pd.date_range(last_date[0], periods=prediction_days, freq="D")
    dfPredict.cost = history[-prediction_days:]
    # dfPredict.set_index("date", inplace=True)

    ## Combine two data lists
    # dfOrginal = dfOrginal.append(dfPredict)

    # plt.plot(dfOrginal.index, dfOrginal)
    # plt.show( )
    
    return dfOrginal, dfPredict

# inverse the difference in the dataset
def inverse_diffference(history, predict, interval=1):
    # print(predict, " + ", history[-interval])
    return predict + history[-interval]

# find the difference in the dataset
def difference(df):
    diff_list = []
    interval = 1

    for i in range(interval, len(df)):
        value = df[i] - df[i - interval]
        # print("i = ", i, " i - interval = ", (i -interval))
        # print(df[i], " - ", df[i - interval], " = ", value)
        diff_list.append(value)

    return array(diff_list)

# convert to number
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

# processing data from psql
def processingDataset(dataset, predict_type):
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

    # today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    # dfDay.index[-1] #yyyy-mm-dd hh:mm:ss

    # Filter costType
    if predict_type == "income":
        incomeList = df[df.costType==0]
        # incomeList = incomeList.drop(columns="costType")
        
        dfDay = insertZero(incomeList)

        return dfDay
    elif predict_type == "expense":
        expensesList = df[df.costType==1]
        # expensesList = expensesList.drop(columns="costType")

        dfDay = insertZero(expensesList)

        return dfDay

    ## Sort by date, item type, and cost type
    # dfGroup = expensesList.groupby(["date", "itemType"]).sum()

    ## Weekly total
    # dfWeek = expensesList.cost.resample('W').sum()
    ## Monthly total
    # dfMonth = expensesList.cost.resample('M').sum()

    # dfWeekDay = pd.merge(dfWeek, dfDay, how="outer", on="date", sort=True, suffixes=("_week", "_day"))
    # dfMonth = pd.merge(dfMonth, dfWeekDay, how="outer", on="date", sort=True)
    # dfMonth = dfMonth.replace(to_replace=np.nan, value=-1)

# predict function
def predict(days, predict_type):
    # Connect to psql server
    engine = create_engine(connectpsql.psql)
    sql_command = "SELECT date, item_type, cost_type, cost FROM budget_item ORDER BY date"

    # Read dataset from psql server
    dataset = pd.read_sql(sql_command, engine, parse_dates=["date"])
    
    # Processing data according to predict type
    dfDay = processingDataset(dataset, predict_type)

    # 5, 1, 0 > 188.426 497.815 (630.170)
    # 5, 0, 1 > 1.549   445.254 564.480
    # 6, 0, 1 > 1.671   438.906 561.012
    # 7, 0, 1 > 1.819   448.877

    days_in_month = 31      # Comparing x days difference
    prediction_days = 14    # Predicting days

    diff_list = difference(dfDay, days_in_month)

    model = ARIMA(diff_list, order=(4, 1, 1))
    model_fit = model.fit(disp=0)

    # start_index = len(diff_list)
    # end_index = start_index + 29
    # forecast = model_fit.predict(start=start_index, end=end_index)
    forecast = model_fit.forecast(steps=days)[0]
    # forecast = inverse_diffference(dfDay, forecast, days_in_month)
    # print("Forecast: %f" % forecast)
    # print(model_fit.forecast(steps=days))
    history = [x for x in dfDay]

    for predict in forecast:
        inverted = inverse_diffference(history, predict, days_in_month)
        if inverted < 0:
            inverted = 0.0
        history.append(round(inverted, 2))

    # find the prediction of how many days
    dfOrginal, dfPredict = processPrediction(dfDay, history, days)

    # convert to html form
    predict_html = dfPredict.to_html(header=False, index_names=False, border=0, classes="predictTable")

    return predict_html, dfOrginal, dfPredict
    # return render(request, "home.html", {"predict_list": predict_html})

# delete item
def deleteItems(request, id):
    Item.objects.filter(id = id).delete()
    return redirect("/home/")

# update item
def updateItems(request, id):
    item = Item.objects.get(id=id)

    if request.method == "POST":
        updateForm = ItemForm(request.POST, instance=item)
        if updateForm.is_valid():
            itemDB = updateForm.save(commit=False)
            itemDB.save()

        return redirect("/home/")
    else:
        updateForm = ItemForm(instance=item)

    return render(request, "updateItems.html", {
            "item":item,
            "updateForm":updateForm
        })


    # if request.method == "POST":
    #     item = request.POST["item"]
    #     item_type = request.POST["item_type"]
    #     cost = request.POST["cost"]
    #     cost_type = request.POST["cost_type"]
    #     remark = request.POST["remark"]
    #     date = request.POST["date"]

    #     Item.objects.filter(id = id).update(
    #         item = item, 
    #         item_type = item_type, 
    #         cost = cost, 
    #         cost_type = cost_type, 
    #         remark = remark,
    #         date = date)

    #     return redirect("home")
    # else:
    #     item = get_object_or_404(Item, id=id)

    #     return render(request, "updateItems.html", {
    #         "item":item
    #     })

# insert item
def insertItems(request):
    if request.method == "POST":
        insertForm = ItemForm(request.POST)
        if insertForm.is_valid():
            item = insertForm.cleaned_data["item"]
            item_type = insertForm.cleaned_data["item_type"]
            cost = insertForm.cleaned_data["cost"]
            cost_type = insertForm.cleaned_data["cost_type"]
            remark = insertForm.cleaned_data["remark"]
            date = insertForm.cleaned_data["date"]

            itemDB = Item(
                item = item, 
                item_type = item_type, 
                cost = cost, 
                cost_type = cost_type, 
                remark = remark,
                date = date,
                username = User.objects.get(username = request.user))

            itemDB.save()

    return redirect("/home/")

# calculate income/expense total
def calculateToTal(request):
    username = Q(username=request.user)

    costtype = Q(cost_type="Expense")
    expense = Item.objects.filter(username, costtype).aggregate(total_sum=Sum("cost"))["total_sum"]
    
    costtype = Q(cost_type="Income")
    income = Item.objects.filter(username, costtype).aggregate(total_sum=Sum("cost"))["total_sum"]

    return income, expense

# main function
def home(request):
    if request.user.is_authenticated:
        items = Item.objects.filter(username = request.user).order_by("-date") # order by date
        itemForm = ItemForm()
        dayForm = DayForm()
        
        income, expense = calculateToTal(request)
        predict_list = ""

        # prediction post request form
        # if request.method == "POST":
        #     form = DayForm(request.POST)
            
        #     if form.is_valid():
        #         predict_type = form.cleaned_data["predict_type"]
        #         days = form.cleaned_data["days"]
        #         print(predict_type)
        #         print(days)
        #         if days == 0:
        #             return redirect("/home/")
        #         else:
        #             predict_list, dfOrginal, dfPredict = predict(days, predict_type)

        #             # plt.plot(dfOrginal.index, dfOrginal, "b-")
        #             # plt.xlabel("Date")
        #             # plt.ylabel("Cost", color="b")
        #             # plt.show()

        return render(request, "home.html", {
            "items": items, 
            "expense": expense,
            "income": income,
            "predict_list": predict_list,
            "item_form": itemForm,
            "day_form": dayForm
            })
    else:
        return redirect("/user/login/")

class DayFormView(AjaxFormMixin, FormView):
    form_class = DayForm
    template_name = "testing.html"
    success_url = "/form-success/"

def testing_data(request):
    if request.method == "POST":
        form = DayForm(request.POST)
        # serializer = DaySerializer(request.POST)

        if form.is_valid():  
            predict_type = form.cleaned_data['predict_type']
            days = form.cleaned_data['days']

            predict_list, dfOrginal, dfPredict = predict(days, predict_type)

            dfOrginal.date = dfOrginal.date.dt.strftime('%Y-%m-%d')
            dfPredict.date = dfPredict.date.dt.strftime('%Y-%m-%d')
            
            data = {
                "predict": dfPredict.cost.to_json(orient="values"),
                "predict_date": dfPredict.date.to_json(orient="values", date_format="iso"),
                "orginal": dfOrginal.cost.to_json(orient="values"),
                "orginal_date": dfOrginal.date.to_json(orient="values", date_format="iso"),
            }
            
            return JsonResponse(data)
        else:
            return JsonResponse(form.errors)
    else:
        return render(request, "testing.html", {
            "testing_data": "No post request!"
            })

# Django REST framework
class PredictChart(AjaxFormMixin, APIView):
    def post(self, request, format=None):
        if request.method == 'POST':
            form = DayForm(request.POST)

            if form.is_valid():  
                data = form.cleaned_data

                # predict_list, dfOrginal, dfPredict = predict(days, predict_type)

                # original = dfOrginal.cost
                # date = dfOrginal.index

                return Response(data)
            else:
                return Response({"Error": "Form failed"})
        else:
            return Response({"Error": "Request Post failed"})
        
# pg_ctl.exe start -D C:\Users\adrian\Apps\PostgreSQL\data
