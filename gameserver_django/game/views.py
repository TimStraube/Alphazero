import json
import random
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def menu(request):
    return render(request, 'menu.html', {}, status=200)

def game(request):
    context = {
        'version': str(random.randint(1, 1000000)),
    }
    return render(request, 'game.html', context, status=200)

def won(request):
    return render(request, 'won.html', {}, status=200)

def loss(request):
    return render(request, 'loss.html', {}, status=200)

def custom_404(request, exception):
    return render(request, '404.html', {}, status=404)