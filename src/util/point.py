"""
description: Defines a Point class representing a point in 2D space.
author: Tim Straube
licence: MIT
"""

class Point():
    X : int = 0
    Y : int = 0
    def __init__(self, x : int, y : int) -> None:
        self.X = x
        self.Y = y

    def X(self):
        return self.X

    def Y(self):
        return self.Y
    
    def exchange_X_value(self, point):
        tmp = point.X
        point.X = self.X
        self.X = tmp
        return [self, point]
    
    def exchange_Y_value(self, point):
        tmp = point.Y
        point.Y = self.Y
        self.Y = tmp
        return [self, point]