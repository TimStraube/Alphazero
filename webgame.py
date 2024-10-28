"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

from game import Battleship
from point import Point

class WebBattleship(Battleship):
    battle = 0
    phase_play = True
    bow = Point(0, 0)

    def __init__(self, size):
        self.size = size
        super().__init__(size)

    def restart(self, player):
        self.battle = 1
        return super().restart(player)

    def bow(self, bow):
        self.bow = bow

    def place_ships_manual(self, state, player, stern):
        failed = 0
        # state, failed, message
        return [
            state, 
            failed, 
            "ship crashed into another one"
        ]
    