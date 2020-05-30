from django.urls import path
from . import views

# home/
urlpatterns = [
    path("", views.home, name="home"),
    path("insertItems", views.insertItems, name="insertItems"),
    path("updateItems/<id>", views.updateItems, name="updateItems"),
    path("deleteItems/<id>", views.deleteItems, name="deleteItems"),
    path("new_insertItems", views.new_insertItems, name="new_insertItems"),
    path("new_updateItems/<id>", views.new_updateItems, name="new_updateItems"),
    path("new_deleteItems/<id>", views.new_deleteItems, name="new_deleteItems"),
    path("testing/chart/data", views.PredictChart.as_view(), name="testingData"),
    path("dataset/", views.requestData, name="requestData"),
    # path("data/", views.DayFormView.as_view(), name="dayformview")
]