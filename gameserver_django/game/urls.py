from django.urls import path
from .views import menu, game, won, loss

urlpatterns = [
    path('', menu, name='menu'),
    path('game/', game, name='game'),
    path('won/', won, name='won'),
    path('loss/', loss, name='loss'),
]

