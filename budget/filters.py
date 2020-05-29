import django_filters
from .models import New_item

class ItemFilters(django_filters.FilterSet):
    item_type_choices = [
        ('Housing', 'Housing'),
        ('Transportation', 'Transportation'),
        ('Utilities', 'Utilities'),
        ('Food', 'Food'),
        ('Entertainment', 'Entertainment'),
        ('Education', 'Education'),
        ('Insurance', 'Insurance'),
        ('Medical/Healthcare', 'Medical/Healthcare'),
        ('Donations', 'Donations'),
        ("Finance Costs", "Finance Costs"),
        ('Others', 'Others')
    ]

    cost_type_choices = [
        ('Income', 'Income'),
        ('Expense', 'Expense')
    ]

    date_choices = [
        ('today', 'Today'),
        ('month', 'Month'),
        ('year', 'Year')
    ]

    item = django_filters.CharFilter(lookup_expr='icontains')
    item_type = django_filters.MultipleChoiceFilter(label='Item Type', choices=item_type_choices)
    cost = django_filters.RangeFilter(widget=django_filters.widgets.RangeWidget(attrs={'type': 'number', 'min': '0', 'step': '.01'}))
    cost_type = django_filters.MultipleChoiceFilter(label='Cost Type', choices=cost_type_choices)
    date = django_filters.DateFromToRangeFilter(widget=django_filters.widgets.RangeWidget(attrs={'class': 'datepicker', 'type': 'date'}))

    class Meta:
        model = New_item
        fields = '__all__'
        exclude = ['remark', 'username']
        