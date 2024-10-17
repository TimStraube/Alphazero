from django.shortcuts import render

def menu(request):
    return render(request, 'menu.html', {}, status=200)

def game(request):
    return render(request, 'game.html', {}, status=200)

def won(request):
    return render(request, 'won.html', {}, status=200)

def loss(request):
    return render(request, 'loss.html', {}, status=200)

def custom_404(request, exception):
    return render(request, '404.html', {}, status=404)