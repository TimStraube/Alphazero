import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from webgameserver.gameserver import Gameserver

gameserver = Gameserver()

def menu(request):
    return render(request, 'menu.html', {}, status=200)

def game(request):
    return render(request, 'game.html', {}, status=200)

def won(request):
    return render(request, 'won.html', {}, status=200)

def loss(request):
    return render(request, 'loss.html', {}, status=200)

@csrf_exempt
def process_svg_source(request):
    # if request.method == 'POST':
    #     data = json.loads(request.body)
    #     source = data.get('Source')

    #     # Hier kannst du die Logik zur Verarbeitung des 'source' hinzufügen
    #     # Beispiel:
    #     state = "new_state"
    #     player = "new_player"
    #     failed = False
    #     message = ""

    #     # Beispiel-Logik zur Bestimmung des Ergebnisses
    #     if source == "winning_move":
    #         message = "You won"
    #     elif source == "losing_move":
    #         message = "AI won"
    #     else:
    #         message = "Continue"

    #     response_data = {
    #         'State': state,
    #         'Player': player,
    #         'Failed': failed,
    #         'Message': message
    #     }

    #     return JsonResponse(response_data)
    # return JsonResponse(
    #     {'error': 'Invalid request'}, 
    #     status=400
    # )

    if request.method == 'POST':
        req = request.get_json()
        json_response = gameserver.singleclick(
            req["Source"]
        )
        return json_response

def custom_404(request, exception):
    return render(request, '404.html', {}, status=404)