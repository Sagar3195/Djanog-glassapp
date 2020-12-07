from django.http import HttpResponse
from django.shortcuts import render
import joblib
def home(request):
    return render(request, 'home.html')

def result(request):
    model = joblib.load("glass_model.pkl")
    lst = []
    lst.append(request.GET['RI'])
    lst.append(request.GET['Na'])
    lst.append(request.GET['Mg'])
    lst.append(request.GET['Al'])
    lst.append(request.GET['Si'])
    lst.append(request.GET['K'])
    lst.append(request.GET['Ca'])
    lst.append(request.GET['Ba'])
    lst.append(request.GET['Fe'])

    ans = model.predict([lst])

    return render(request, 'result.html', {'ans': ans})

