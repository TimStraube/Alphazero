"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import tensorflow as tf
import random
import numpy
from agents.alphazero.residualnetwork import ResidualNetwork
from envs.battleship import Battleship
from src.agents.alphazero.mcts import MCTS

def quantize_model(converter):
    board_size = 9
    game = Battleship(board_size)   
    state = game.restart(0)
    model = ResidualNetwork(
                game, 
                4, 
                9, 
                9, 
                device = 'cpu'
            )
    random_number_moves = random.randint(
        28, 
        2 * board_size * board_size
    )
    args = {
        'C': 2,
        'num_searches': 100,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1
    }
    mcts = MCTS(game, args, model)
    player = -1

    def representative_data_gen(game, state, mcts, player, random_number_moves):
        for _ in range(random_number_moves):
            neutral_state = game.change_perspective(
                game.get_encoded_state(state), player
            )
            mcts_probs = mcts.search(neutral_state)
            action = numpy.argmax(mcts_probs)
            state = game.step(
                game.get_encoded_state(state), 
                action, 
                player
            )
            player = -player
            yield state

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen(game,state, mcts, player, random_number_moves)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,        
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.   
        ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    return converter.convert()


saved_model_dir = '.'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

tflite_model = quantize_model(converter)
open("vivalavida.tflite", "wb").write(tflite_model)
