import numpy as np
import random

state = np.zeros((6, 9, 9), dtype=np.uint8)
ships_possible = [[5, 4, 3, 2], [5, 4, 3, 2]]
num_shipparts = 14
size = 9

def shipIndex(player):
    return 3 * int(player > 0)

def fcn_set_ship(state, player):
    # TODO test
    for ship in ships_possible[int(player > 0)]:
        random_direction = random.randint(0, 1)
        positions = np.array([])

        # loop for checking if a ship can be placed
        for i in range(size - ship + 1):
            prefix = np.zeros((1, i), dtype=np.uint8)
            body = np.ones((1, ship), dtype=np.uint8)
            postfix = np.zeros((1, size - ship - i), dtype=np.uint8)
            ship_possible = np.concatenate((prefix, body, postfix), axis=1)
            if random_direction:
                ship_possible_squeezed = np.squeeze(np.matmul(
                    ship_possible,
                    np.logical_not(state[shipIndex(player), :, :])) == ship)
            else:
                ship_possible_squeezed = np.squeeze(np.matmul(
                    ship_possible,
                    np.logical_not(
                        np.transpose(state[shipIndex(player), :, :]))) == ship)
            positions = np.append(
                positions,
                ship_possible_squeezed,
                axis=0)

        positions = np.reshape(positions, (size - ship + 1, size))
        possible_positions = np.where(positions == 1)

        length_possible_positions = possible_positions[0].size

        random_ship_position = random.randint(
            0, 
            length_possible_positions - 1)

        x = possible_positions[0][random_ship_position]
        y = possible_positions[1][random_ship_position]

        if random_direction:
            p1 = [x, y]
            p2 = [x + ship - 1, y]
        else:
            p1 = [y, x]
            p2 = [y, x + ship - 1]

        ship_array = fcn_get_points_between(p1, p2)

        for point in ship_array:
            state[shipIndex(player), point[0], point[1]] = 255

def fcn_get_points_between(p1, p2):
    points = []

    if p1[0] == p2[0]:  # If x1 and x2 are the same
        y_values = list(range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1))
        points = [[p1[0], y] for y in y_values]
    elif p1[1] == p2[1]:  # If y1 and y2 are the same
        x_values = list(range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1))
        points = [[x, p1[1]] for x in x_values]

    return points

fcn_set_ship(state, 1)
fcn_set_ship(state, -1)

print(state)

