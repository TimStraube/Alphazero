"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import numpy as np
import torch
import socket
import torch
import numpy
import time
from django.http import JsonResponse
from webgameserver.residualnetwork import ResidualNetwork
from webgameserver.mcts import MCTS
from gtts import gTTS
import onnxruntime as rt
# from playsound import playsound
from webgameserver.webgame import WebBattleship
from flask import Flask
from flask import render_template
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType

app = Flask(__name__)

class Gameserver():
    game_number = 1
    
    def __init__(self) -> None:
        self.restart()

    def restart(self, num_searches = 100):
        self.webgame = WebBattleship(9)
        self.player = -1
        self.moves = 0
        self.next_round_no_ai = False
        self.game_over_enable = True
        args = {
            'C': 2,
            'num_searches': num_searches,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.1
        }
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = "AZ-24-06-24-EIPC51-A"

        self.model = ResidualNetwork(
            self.webgame, 
            2, 
            32, 
            4, 
            device
        )
        self.model.load_state_dict(
            torch.load(
                f"./webgameserver/models/{self.model_path}/main.pt", 
                map_location = device
            )
        )
        self.model.eval()
        self.mcts = MCTS(self.webgame, args, self.model)
        self.state = self.webgame.restart(self.player)
        self.game_number = self.game_number + 1

        model_path = (
            f"./webgameserver/models/{self.model_path}/model_quantized.onnx"
        )
       
        self.sess = rt.InferenceSession(
            model_path
        )

    def singleclick(self, action):
        self.webgame.repeat = True
        message = ""

        if not self.next_round_no_ai:
            while self.webgame.repeat == True:
                neutral_state = self.webgame.change_perspective(
                    self.state.copy(), 
                    self.player
                )

                input_name = self.sess.get_inputs(
                )[0].name
                # print("input name", input_name)
                input_shape = self.sess.get_inputs(
                )[0].shape
                # print("input shape", input_shape)

                start = time.time()

                mcts_probs = self.mcts.search(neutral_state)
                
                end = time.time()
                
                # print(f"Time: {end - start}")

                # print(f"mcts: {mcts_probs}")
                action_ai = np.argmax(mcts_probs)
                self.state = self.webgame.step(
                    self.state, 
                    action_ai,
                    self.player
                )

        action_human = (
            self.webgame.size * 
            (int(ord(action[0])) - 65) + 
            (int(action[1]) - 1)
        )
        self.state = self.webgame.step(
            self.state, 
            action_human, 
            -self.player
        )

        self.moves = self.moves + 1

        if self.webgame.check_win(
            self.state, 
            action_human, 
            1):

            print(f"Moves human to win: {self.moves}")
            if self.game_over_enable:
                message = "You won"
                
        if not self.next_round_no_ai:
            if self.webgame.check_win(
                self.state, 
                action_ai, 
                -1):

                print(f"Moves ai to win: {self.moves}")
                if self.game_over_enable:
                    message = "AI won"

        if self.webgame.repeat:
            self.next_round_no_ai = True
        else:
            self.next_round_no_ai = False

        json_message = {
            "State" : self.state.tolist(),
            "Player" : self.player,
            "Failed" : 0,
            "Message" : message
        }
        
        # gtts_message = gTTS(message, lang="de", slow=False)
        # gtts_message.save("sample.mp3")
        # playsound("sample.mp3")
        return JsonResponse(json_message)

    # handle actions for game with human and computer agent
    def doubleclick(self, action):
        if self.player == -1:
            neutral_state = self.webgame.change_perspective(
                self.state.copy(), 
                self.player
            )
            mcts_probs = self.mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            message = "Step by ai"
            if self.webgame.check_win(
                self.state, 
                action, 
                -1):

                if self.game_over_enable:
                    message = "AI won"
            state = self.webgame.step(
                self.state, 
                action, 
                self.player
            )
        else:
            action = (
                self.webgame.size * 
                (int(ord(action[0])) - 65) + 
                int(action[1]) - 1
            )
            message = "Step by human"
            if self.webgame.check_win(
                self.state, 
                action, 
                1):

                if self.game_over_enable:
                    message = "You won"
            state = self.webgame.step(
                self.state, 
                action, 
                self.player
            )

        json_message = JsonResponse({
            "State" : state.tolist(),
            "Player" : self.player,
            "Failed" : 0,
            "Message" : message
        })
        if not self.webgame.repeat:
            self.player = -self.player
        # gtts_message = gTTS(message, lang="de", slow=False)
        # gtts_message.save("sample.mp3")
        # playsound("sample.mp3")
        return json_message
        
    def fcn_get_moves_array(self):
        json_message = JsonResponse({'Moves' : self.moves_array})
        return json_message

gameserver = Gameserver()

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

if __name__ == "__main__":
    print("Welcome to Sink Ships :)")
    hostname = socket.gethostbyname(socket.gethostname())
    app.run(debug = False, host = hostname)