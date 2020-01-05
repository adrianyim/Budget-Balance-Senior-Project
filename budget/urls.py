from django.urls import path
from . import views

# home/
urlpatterns = [
    path("", views.home, name="home"),
    path("insertItems", views.insertItems, name="insertItems"),
    path("updateItems/<id>", views.updateItems, name="updateItems"),
    path("deleteItems/<id>", views.deleteItems, name="deleteItems")
]