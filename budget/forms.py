from django import forms
from .models import Item

class DateInput(forms.DateInput):
    input_type = "date"

class ItemForm(forms.ModelForm):
    cost = forms.DecimalField(min_value=0)
    remark = forms.CharField(widget=forms.Textarea(attrs={"rows":2, "cols":25}), required=False)

    class Meta:
        model = Item
        fields = ["item", "item_type", "cost", "cost_type", "remark", "date"]
        itemType_Choices = [
            ("", "--Item Type--"),
            ("Salaries and wages", "Salaries and wages"),
            ("Utility expenses", "Utility expenses"),
            ("Administration expenses", "Administration expenses"),
            ("Finance costs", "Finance costs"),
            ("Depreciation", "Depreciation"),
            ("Impairment losses", "Impairment losses"),
            ("Food", "Food"),
            ("Others", "Others")
            ]
        costType_Choices = [
            ("", "--Cost Type--"),
            ("Income", "Income"),
            ("Expense", "Expense")
        ]
        widgets = {
            "item_type": forms.Select(choices=itemType_Choices),
            "cost_type": forms.Select(choices=costType_Choices),
            "date": DateInput()
        }

class DayFrom(forms.Form):
    predict_type = forms.ChoiceField(choices=[
        ("income", "Income"),
        ("expense", "Expense")
    ])
    days = forms.IntegerField(min_value=0)