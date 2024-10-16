from django.urls import path
from .views import menu, game

urlpatterns = [
    path('', menu, name='menu'),
    path('game/', game, name='game'),
]

