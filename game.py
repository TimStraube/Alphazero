import numpy
import random

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        self.moves = 0

    def __repr__(self):
        return "connectfour"

    def get_initial_state(self):
        return numpy.zeros(
            (self.row_count, self.column_count)
        )

    def step(self, state, action, player):
        row = numpy.max(numpy.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(numpy.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = numpy.min(numpy.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1  
        )

    def terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if numpy.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state, player):
        state = state[player:player + 2]
        encoded_state = numpy.stack(
            (state == 0, state == 255)
        ).astype(numpy.float32)
        return encoded_state

class Battleship:
    def __init__(self, size):
        # player 0 and 3 as indices for map 
        self.row_count = size
        self.column_count = size
        self.size = size
        self.action_size = (
            self.column_count * self.row_count
        )
        self.moves = 0

    def __repr__(self):
        return "battleship"

    def restart(self, player):
        self.repeat = False
        self.ships_possible = [[5, 4, 3, 2], [5, 4, 3, 2]]
        self.num_shipparts = 14
        # initalization of all submaps
        state = numpy.zeros(
            (6, self.column_count, self.row_count), dtype=numpy.uint8
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

        hit = state[self.hitIndex(player), x, y]
        ship = state[self.shipIndex(-player), x, y]

        self.repeat = False

        if (hit == 0 and ship == 0):
            # hit water
            state[self.hitIndex(player), x, y] = 255
        elif (hit == 0 and ship == 255):
            # hit ship
            state[self.hitIndex(player), x, y] = 255
            state[self.knowledgeIndex(player), x, y] = 255
            # for ii in range(len(self.ships[int(player > 0)])):
            #     self.ships[int(player > 0)][ii].remove([x, y])
            self.repeat = True
        else:
            pass
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
        hit_state = state[self.hitIndex(player)]
        ship_state = state[self.shipIndex(-player)]
        if (numpy.sum(ship_state * hit_state) == 
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
            (6, self.column_count, self.row_count), 
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
            random_direction = random.randint(0, 1)

            # random_direction = 0

            positions = numpy.array([])

            # loop for checking if a ship can be placed
            for i in range(self.size - ship + 1):
                prefix = numpy.zeros(
                    (1, i), 
                    dtype = numpy.uint8
                )
                body = numpy.ones(
                    (1, ship), 
                    dtype = numpy.uint8
                )
                postfix = numpy.zeros(
                    (1, self.size - ship - i), 
                    dtype = numpy.uint8
                )
                ship_possible = numpy.concatenate(
                    (prefix, body, postfix), 
                    axis = 1
                )
                if random_direction:
                    ship_possible_squeezed = numpy.squeeze(
                        numpy.matmul(
                            ship_possible,
                            numpy.logical_not(
                                state[
                                    self.shipIndex(player), :, 
                                    :
                                ]
                            )
                        ) == ship
                    )
                else:
                    transposed_shipmap = numpy.transpose(
                        state[
                            self.shipIndex(player), 
                            :, 
                            :
                        ]
                    )
                    ship_possible_squeezed = numpy.squeeze(
                        numpy.matmul(
                            ship_possible,
                            numpy.logical_not(
                                transposed_shipmap
                            )
                        ) == ship
                    )
                positions = numpy.append(
                    positions,
                    ship_possible_squeezed,
                    axis = 0
                )

            positions = numpy.reshape(
                positions, 
                (self.size - ship + 1, self.size)
            )
            possible_positions = numpy.where(positions == 1)

            length_possible_positions = possible_positions[0].size

            random_ship_position = random.randint(
                0, 
                length_possible_positions - 1
            )

            x = possible_positions[0][random_ship_position]
            y = possible_positions[1][random_ship_position]

            if random_direction:
                p1 = [x, y]
                p2 = [x + ship - 1, y]
            else:
                p1 = [y, x]
                p2 = [y, x + ship - 1]

            ship_array = self.points_between(p1, p2)

            self.ships[int(player > 0)].append(ship_array)

            for point in ship_array:
                state[
                    self.shipIndex(player), 
                    point[0], 
                    point[1]
                ] = 255

    def points_between(self, p1, p2):
        points = []

        if p1[0] == p2[0]:  # If x1 and x2 are the same
            y_values = list(
                range(
                    min(p1[1], p2[1]), 
                    max(p1[1], p2[1]) + 1
                )
            )
            points = [[p1[0], y] for y in y_values]
        elif p1[1] == p2[1]:  # If y1 and y2 are the same
            x_values = list(
                range(
                    min(p1[0], p2[0]), 
                    max(p1[0], p2[0]) + 1
                )
            )
            points = [[x, p1[1]] for x in x_values]

        return points
