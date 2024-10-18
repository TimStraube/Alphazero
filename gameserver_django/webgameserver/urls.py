from django.urls import path
from .views import menu, game, won, loss, process_svg_source

urlpatterns = [
    path('', menu, name='menu'),
    path('game/', game, name='game'),
    path('won/', won, name='won'),
    path('loss/', loss, name='loss'),
    path('process_svg_source/', process_svg_source, name='process_svg_source'),
]

