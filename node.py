import math
import numpy

class Node:
    def __init__(
        self, 
        game, 
        args, 
        state, 
        parent=None, 
        action_taken=None, 
        prior=0, 
        visit_count=0):
        
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.prior = prior
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        # return numpy.sum(self.expandable_moves) == 0 and len(self.children) > 0
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -numpy.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((
                child.value_sum / 
                child.visit_count
            ) + 1) / 2
        # return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        return (
            q_value + 
            self.args['C'] * 
            (
                math.sqrt(self.visit_count) / 
                (child.visit_count + 1)
            ) * 
            child.prior
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.step(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(
                    self.game,
                    self.args,
                    child_state,
                    self, 
                    action, 
                    prob)
                self.children.append(child)
        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)