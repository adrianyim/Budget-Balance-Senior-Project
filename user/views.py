from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.contrib.auth import get_user_model

User = get_user_model()

# Create your views here.
def logout(request):
    auth.logout(request)
    return redirect("login")

def login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = auth.authenticate(
            username = username, 
            password = password)

        if user is not None:
            auth.login(request, user)
            return redirect("/home")
        else:
            messages.info(request, "Username or password is incorrected")
            return redirect("login")
    else:
        return render(request, "login.html")

def register(request):
    if request.method == "POST":
        first_name = request.POST["first_name"]
        last_name = request.POST["last_name"]
        username = request.POST["username"]
        password = request.POST["password"]
        repassword = request.POST["repassword"]

        if User.objects.filter(username = username).exists():
            messages.info(request, "Username is existed")
            return redirect("register")
        elif password == repassword:
            user = User.objects.create_user(
                first_name = first_name, 
                last_name = last_name, 
                username = username, 
                password = password)
            user.save()
            return redirect("login")
        else:
            messages.info(request, "Password is not matching")
            return redirect("register")
    else:
        return render(request, "register.html")