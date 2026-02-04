"""
description: Implementation of the Battleship game environment.
author: Tim Straube
licence: MIT
"""

import numpy
import random

class Battleship:
    def __init__(self, size, debug=False):
        # player 0 and 3 as indices for map 
        self.rows = size
        self.columns = size
        self.size = size
        self.actions = (
            self.columns * self.rows
        )
        self.moves = 0
        self.debug = debug

    def __repr__(self):
        return "battleship"

    def restart(self, player):
        self.repeat = False
        self.ships_possible = [[3, 2], [3, 2]]
        self.num_shipparts = sum(self.ships_possible[0])
        # initalization of all submaps
        state = numpy.zeros(
            (6, self.columns, self.rows), dtype=numpy.uint8
        )
        self.ships = [[], []]
        self.place_ships(state, player)
        self.place_ships(state, -player)
        return state
    
    def shipIndex(self, player):
        # f: {-1, 1} -> {0, 3}
        return 3 * int(player > 0)
    
    def hitIndex(self, player):
        # f: {-1, 1} -> {1, 4}
        return 3 * int(player > 0) + 1
    
    def knowledgeIndex(self, player):
        # f: {-1, 1} -> {2, 5}
        return 3 * int(player > 0) + 2

    def step(self, state, action, player):
        x = action // self.size
        y = action % self.size

        hit = int(state[self.hitIndex(player), x, y])
        ship = int(state[self.shipIndex(-player), x, y])

        self.repeat = False

        # Explicit numeric checks (0 == unknown, 255 == marked)
        if hit == 0 and ship == 0:
            # hit water
            state[self.hitIndex(player), x, y] = 255
            if self.debug:
                print(f"Battleship.step: water at {(x,y)} by player {player}")
        elif hit == 0 and ship == 255:
            # hit ship
            state[self.hitIndex(player), x, y] = 255
            state[self.knowledgeIndex(player), x, y] = 255
            # note: we keep ship bookkeeping but do not remove parts here
            self.repeat = True
            if self.debug:
                print(f"Battleship.step: HIT at {(x,y)} by player {player} (ship layer index {self.shipIndex(-player)})")
        else:
            # already hit before or other state; do nothing
            if self.debug:
                print(f"Battleship.step: no-op at {(x,y)} hit={hit} ship={ship} player={player}")
        return state

    def get_valid_moves(self, state, player):
        return (
            (state[self.hitIndex(player), :, :] == 0)
            .astype(numpy.uint8)
            .flatten()
        )
    
    def policy(self, policy, state):
        valid_moves = (
            (state[self.hitIndex(1), :, :] == 0)
            .astype(numpy.uint8)
            .flatten()
        )
        policy *= valid_moves
        policy /= numpy.sum(policy)
        return policy

    def check_win(self, state, action, player):
        state_hit = state[self.hitIndex(player)]
        state_ship = state[self.shipIndex(-player)]
        if (numpy.sum(state_ship * state_hit) == 
            self.num_shipparts):
            
            return True
        else:
            return False

    def terminated(self, state, action):
        if self.check_win(state, action, 1):
            return 1, True
        if self.check_win(state, action, -1):
            return 1, True
        return 0, False

    def change_perspective(self, state, player):
        # TODO test
        return_state = numpy.zeros(
            (6, self.columns, self.rows), 
            dtype=numpy.uint8
        )
        if player == -1:
            state_copy = state[0:3]
            return_state[0:3] = state[3:6]
            return_state[3:6] = state_copy
            return return_state
        else:
            return state

    def get_encoded_state(self, state):
        obsA = (
            state[
                self.hitIndex(1) : 
                self.knowledgeIndex(1) + 1
            ] == 255
        ).astype(numpy.float32)
        obsB = (
            state[
                self.hitIndex(-1) : 
                self.knowledgeIndex(-1) + 1
            ] == 255
        ).astype(numpy.float32)
        observation = numpy.concatenate(
            (obsB, obsA), 
            axis=0
        )
        # observation = (state == 255).astype(numpy.float32)
        return observation

    def place_ships(self, state, player):
        for ship in self.ships_possible[int(player > 0)]:
            max_attempts = 1000
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1
                random_direction = random.randint(0, 1)
                possible_positions = []

                # horizontal placement: x varies, y constant
                if random_direction == 1:
                    for x in range(0, self.size - ship + 1):
                        for y in range(0, self.size):
                            segment = [state[self.shipIndex(player), x + i, y] for i in range(ship)]
                            if all(int(v) == 0 for v in segment):
                                possible_positions.append(((x, y), (x + ship - 1, y)))
                else:
                    # vertical placement: y varies, x constant
                    for y in range(0, self.size - ship + 1):
                        for x in range(0, self.size):
                            segment = [state[self.shipIndex(player), x, y + i] for i in range(ship)]
                            if all(int(v) == 0 for v in segment):
                                possible_positions.append(((x, y), (x, y + ship - 1)))

                if not possible_positions:
                    # try another random orientation / attempt
                    continue

                p1, p2 = random.choice(possible_positions)
                ship_array = self.points_between([p1[0], p1[1]], [p2[0], p2[1]])
                self.ships[int(player > 0)].append(ship_array)
                for point in ship_array:
                    state[self.shipIndex(player), point[0], point[1]] = 255
                placed = True

            if not placed:
                raise ValueError(f"Could not place ship of length {ship} for player {player} after {max_attempts} attempts")

    def points_between(self, p1, p2):
        points = []

        if p1[0] == p2[0]:  
            y_values = list(
                range(
                    min(p1[1], p2[1]), 
                    max(p1[1], p2[1]) + 1
                )
            )
            points = [[p1[0], y] for y in y_values]
        elif p1[1] == p2[1]:  
            x_values = list(
                range(
                    min(p1[0], p2[0]), 
                    max(p1[0], p2[0]) + 1
                )
            )
            points = [[x, p1[1]] for x in x_values]

        return points
