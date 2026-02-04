"""
description: Monte Carlo Tree Search implementation for AlphaZero
secondary author: Tim Straube
licence: MIT
"""

import torch
import numpy
from agents.alphazero.node import Node

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(
            self.game, 
            self.args, 
            state, 
            visit_count = 1
        )
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state), 
                device = self.model.device
            ).unsqueeze(0)
        )
        policy = (
            torch.softmax(policy, axis = 1)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        policy = (
            (1 - self.args['dirichlet_epsilon']) * 
            policy + 
            self.args['dirichlet_epsilon'] * 
            numpy.random.dirichlet(
                [self.args['dirichlet_alpha']] * 
                self.game.actions))
        policy = self.game.policy(policy, state)
        root.expand(policy)
        for search in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()
            value, is_terminal = self.game.terminated(
                node.state,
                node.action_taken
            )
            value = -value
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(
                            node.state
                        ),
                        device = self.model.device
                    ).unsqueeze(0)
                )
                policy = torch.softmax(
                    policy, 
                    axis=1
                ).squeeze(0).detach().cpu().numpy()
                value = value.item()
                policy = self.game.policy(
                    policy, 
                    node.state
                )
                node.expand(policy)
            node.backpropagate(value)
        action_probs = numpy.zeros(self.game.actions)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= numpy.sum(action_probs)
        return action_probs
    
    @torch.no_grad()
    def search_int8(self, state, sess):
        root = Node(
            self.game, 
            self.args, 
            state, 
            visit_count = 1
        )        
        input_name = sess.get_inputs()[0].name
         
        input_data = self.game.get_encoded_state(state)
        input_data = numpy.expand_dims(input_data, axis = 0)
        policy, _ = sess.run(
            None, 
            {input_name: input_data}
        )
        policy = torch.tensor(policy)

        policy = (
            torch.softmax(policy, axis = 1)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        policy = (
            (1 - self.args['dirichlet_epsilon']) * 
            policy + 
            self.args['dirichlet_epsilon'] * 
            numpy.random.dirichlet(
                [self.args['dirichlet_alpha']] * 
                self.game.action_size
            )
        )
        policy = self.game.policy(policy, state)
        root.expand(policy)
        for search in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()
            value, is_terminal = self.game.terminated(
                node.state,
                node.action_taken
            )
            value = -value
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(
                            node.state
                        ),
                        device = self.model.device
                    ).unsqueeze(0)
                )
                policy = torch.softmax(
                    policy, 
                    axis=1
                ).squeeze(0).detach().cpu().numpy()
                value = value.item()
                policy = self.game.policy(
                    policy, 
                    node.state
                )
                node.expand(policy)
            node.backpropagate(value)
        action_probs = numpy.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = (
                child.visit_count
            )
        action_probs /= numpy.sum(action_probs)
        return action_probs