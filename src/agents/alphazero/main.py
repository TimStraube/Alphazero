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
import torch.nn.functional as functional
import concurrent.futures
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from agents.alphazero.mcts import MCTS
from agents.alphazero.residualnetwork import ResidualNetwork
from envs.battleship import Battleship

class AlphaZero:
    def __init__(self):
        print("\nSetup of AlphaZero for training battleship\n")
        model_id = input("ID (string) [alphazero]: ") or "alphazero"
        size = int(input("Battleship board whereas width equals height min 3 (int) [5]: ") or 5)
        if size < 3:
            size = 3
        self.game = Battleship(size)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        resblocks = int(input("Resblocks (int) [6]: ") or 6)
        hiddenlayers = int(input("Hidden layers (int) [6]: ") or 6)
        inputarrays = 4 # int(input("Observed arrays (int): "))
        searches = int(input("Searches (int) [4]: ") or 4)
        selfplayiterations = int(
            input("Self play iterations (int) [64]: ") or 64
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
            'num_epochs': 128,
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

        # TensorBoard writer for training metrics — create per-run subfolder with timestamp
        base_log = os.path.join("logs", model_id)
        try:
            os.makedirs(base_log, exist_ok=True)
        except Exception:
            pass
        try:
            from datetime import datetime
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        except Exception:
            ts = "run"
        run_name = f"{model_id}_{ts}"
        run_dir = os.path.join(base_log, run_name)
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception:
            pass
        try:
            self.writer = SummaryWriter(log_dir=run_dir)
        except Exception:
            self.writer = None
        # create model directory matching the run timestamp
        try:
            model_base = os.path.join("models", model_id)
            os.makedirs(model_base, exist_ok=True)
        except Exception:
            model_base = os.path.join("models", model_id)
        self.model_dir = os.path.join(model_base, run_name)
        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception:
            pass
        self.batch_step = 0
        self.current_iteration = 0
        self.play_step = 0
        self.global_episode_lengths = []

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
                print(f"\nEpisodes: {episodes}")
                # append to per-group and global lists
                self.array_episodes.append(episodes)
                self.global_episode_lengths.append(episodes)
                # log per-episode length to TensorBoard
                try:
                    if self.writer is not None:
                        self.writer.add_scalar('alphazero/episode_length', episodes / 2, self.play_step)
                except Exception:
                    pass
                # increment global episode counter
                self.play_step += 1
                # compute group average and reset group array if full
                l = len(self.array_episodes)
                if l == self.args['num_selfPlay_iterations']:
                    self.average_episodes = sum(self.array_episodes) / len(self.array_episodes)
                    print("\navg episodes " + str(self.average_episodes))
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
            state, policy_targets, value_targets = zip(
                *sample
            )
            state, policy_targets, value_targets = (
                numpy.array(state),
                numpy.array(policy_targets),
                numpy.array(value_targets).reshape(-1, 1)
            )
            state = torch.tensor(
                state, 
                dtype=torch.float32,
                device=self.model.device
            )
            policy_targets = torch.tensor(
                policy_targets,
                dtype=torch.float32,
                device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, 
                dtype=torch.float32, 
                device=self.model.device
            )

            out_policy, out_value = self.model(state)

            policy_loss = functional.cross_entropy(
                out_policy, 
                policy_targets
            )
            value_loss = functional.mse_loss(
                out_value, 
                value_targets
            )
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Log losses to TensorBoard
            try:
                if self.writer is not None:
                    self.writer.add_scalar('alphazero/policy_loss', float(policy_loss.item()), self.batch_step)
                    self.writer.add_scalar('alphazero/value_loss', float(value_loss.item()), self.batch_step)
                    self.writer.add_scalar('alphazero/total_loss', float(loss.item()), self.batch_step)
                    self.batch_step += 1
            except Exception:
                pass

    def learn(self, modellocation):
        for iter_idx in range(self.args['num_iterations']):
            self.current_iteration = iter_idx
            memory = []
            self.model.eval()
            for _ in trange(
                self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            # save model into per-run model dir
            try:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_dir, "main.pt")
                )
            except Exception:
                torch.save(self.model.state_dict(), "./models/" + modellocation + f"/main.pt")

if __name__ == "__main__":
    alphazero = AlphaZero()