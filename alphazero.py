"""
secondary author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import random
import numpy
import os
import sqlite3
import torch
import torch.nn.functional as F
from mcts import MCTS
from tqdm import trange
from game import Battleship
from residualnetwork import ResidualNetwork

class AlphaZero:
    def __init__(self):
        print("\nSetup of AlphaZero for training battleship\n")
        model_id = str(input("ID (string, for example AZ-{day}-{month}-{year}-{hardware}-{char}): "))
        size = int(input("Battleship board width/height (int): "))
        self.game = Battleship(size)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        resblocks = int(input("Resblocks (int): "))
        hiddenlayers = int(input("Hidden layers (int): "))
        inputarrays = int(input("Observed arrays (int): "))
        searches = int(input("Searches (int): "))
        selfplayiterations = int(
            input("Self play iterations (int): ")
        )
        self.model = ResidualNetwork(
            self.game, 
            resblocks, 
            hiddenlayers, 
            inputarrays, 
            device
        )
        # model.load_state_dict(torch.load(
        #     "./projects/rl_battleship/alphazero/models/vivalavida9x9/main.pt", 
        #     map_location=device
        # ))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )
        self.args = {
            'C': 2,
            'num_searches': searches,
            'num_iterations': 256,
            'num_selfPlay_iterations': selfplayiterations,
            'num_epochs': 4,
            'batch_size': 1024,
            'temperature': 1, # 1.25 (nicht verwendet, da die aufsummierten Wahrscheinlichkeiten 1 ergeben müssen)
            'dirichlet_epsilon': 0.25,
            'dirichlet_alpha': 0.3
        }
        try: 
            os.makedirs(os.path.join(
                "./models/", 
                model_id
            ), exist_ok = True)
        except OSError as error:
            pass

        conn = None
        try:
            # Ensure the models folder exists
            os.makedirs("models", exist_ok=True)

            # Path to the database file
            database_path = os.path.join(
                "models", 
                "config.db"
            )

            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            # Create table with name based on model_id
            table_name = f"model_{model_id.replace('-', '_')}" 
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    Size INT,
                    Resblocks INT,
                    Hidden_layer INT,
                    Input_array INT,
                    Searches INT,
                    Selfplay_iterations INT
                )
            """
            cursor.execute(create_table_query)
            
            # Insert model metadata and file path
            insert_query = f"""
                INSERT INTO {table_name} (
                    Size,
                    Resblocks,
                    Hidden_layer,
                    Input_array,
                    Searches,
                    Selfplay_iterations
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            cursor.execute(
                insert_query, 
                (
                    size,
                    resblocks,
                    hiddenlayers,
                    inputarrays,
                    searches,
                    selfplayiterations
                )
            )
            
            # Commit the transaction
            conn.commit()
        except sqlite3.Error as e:
            print("SQLite error: " + str(e))

        self.average_episodes = 0
        self.array_episodes = []
        self.mcts = MCTS(self.game, self.args, self.model)

        self.learn(model_id)

    def selfPlay(self):
        memory = []
        player = 1 
        state = self.game.restart(player)
        episodes = 0

        while True:
            neutral_state = self.game.change_perspective(
                state.copy(), 
                player
            )
            action_probs = self.mcts.search(neutral_state)
            memory.append((
                neutral_state, 
                action_probs, 
                player
            ))
            action = numpy.random.choice(
                self.game.actions, 
                p = action_probs
            )

            state = self.game.step(state, action, player)
            episodes += 1
            value, is_terminal = self.game.terminated(
                state,
                action
            )
            if is_terminal:
                self.array_episodes.append(episodes)
                l = len(self.array_episodes)
                if (l == 
                    self.args['num_selfPlay_iterations']):
                    self.average_episodes = sum(
                        self.array_episodes
                    ) / len(self.array_episodes)
                    print(
                        "\navg episodes " +
                        str(self.average_episodes)
                    )
                    self.array_episodes = []
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    if hist_player == player:
                        hist_outcome = value
                    else:
                        hist_outcome = -value
                    returnMemory.append((
                        self.game.get_encoded_state(
                            hist_neutral_state
                        ),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            player = -player

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[
                batchIdx:min(len(memory) - 1,
                batchIdx + self.args['batch_size'])]  
            # Change to in case of an error
            # memory[batchIdx:batchIdx + self.args['batch_size']] 
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = (
                numpy.array(state),
                numpy.array(policy_targets),
                numpy.array(value_targets).reshape(-1, 1))
            state = torch.tensor(
                state, 
                dtype=torch.float32,
                device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets,
                dtype=torch.float32,
                device=self.model.device)
            value_targets = torch.tensor(
                value_targets, 
                dtype=torch.float32, 
                device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(
                out_policy, 
                policy_targets
            )
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        pass

    def learn(self, modellocation):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for selfPlay_iteration in trange(
                self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            torch.save(
                self.model.state_dict(), 
                "./models/" + modellocation + f"/main.pt"
            )
            # torch.save(
            #     self.optimizer.state_dict(), 
            #     "/home/ti741str/Dokumente/Projektarbeit/ai-systems-lab/projects/rl_battleship/alphazero/connect4_and_battleship/models/" + modellocation + f"optimizer_{iteration}_{self.game}.pt")

if __name__ == "__main__":
    alphazero = AlphaZero()