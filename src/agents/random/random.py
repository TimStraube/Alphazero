import random
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from envs.battleship import Battleship

class RandomAgent:
    def __init__(self, model_id="random", size=5, episodes=100):
        self.model_id = model_id
        self.size = int(size)
        self.episodes = int(episodes)
        if SummaryWriter is not None:
            try:
                self.writer = SummaryWriter(log_dir=f"./logs/{model_id}")
            except Exception:
                self.writer = None
        else:
            self.writer = None

    def play_one(self):
        game = Battleship(self.size)
        player = 1
        state = game.restart(player)
        steps = 0
        while True:
            # choose only valid moves to avoid no-op repeats
            valid_mask = game.get_valid_moves(state, player)
            choices = np.flatnonzero(valid_mask)
            if choices.size == 0:
                # no valid moves left (shouldn't normally happen)
                return steps
            action = int(np.random.choice(choices))
            state = game.step(state, action, player)
            steps += 1
            value, is_terminal = game.terminated(state, action)
            if is_terminal:
                return steps
            player = -player

    def run(self):
        lengths = []
        for i in range(self.episodes):
            l = self.play_one()
            lengths.append(l)
            print(f"Episode {i+1}/{self.episodes}: {l} actions")
            if self.writer is not None:
                try:
                    self.writer.add_scalar("random/episode_length", l, i)
                    # log running average up to this episode (so avg_episodes appears over time)
                    avg_so_far = sum(lengths) / len(lengths)
                    self.writer.add_scalar("random/avg_episodes", float(avg_so_far), i)
                except Exception:
                    pass
        avg = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"avg episodes {avg}")
        if self.writer is not None:
            try:
                # also persist final average at the final step index
                self.writer.add_scalar("random/avg_episodes", avg, len(lengths)-1 if lengths else 0)
                self.writer.flush()
            except Exception:
                pass
        return avg