from django.db import models
from django.conf import settings

# Create your models here.

class Item(models.Model):
    item = models.CharField(max_length=50)
    item_type = models.CharField(max_length=50)
    cost = models.DecimalField(max_digits=10, decimal_places=2)
    cost_type = models.CharField(max_length=10)
    remark = models.TextField()
    date = models.DateField()
    username = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)