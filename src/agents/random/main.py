import random
import numpy as np
import os
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        SummaryWriter = None

from envs.battleship import Battleship

class RandomAgent:
    def __init__(self, model_id="random", size=5, episodes=100, window: int = 100):
        self.model_id = model_id
        self.size = int(size)
        self.episodes = int(episodes)
        self.window = int(window)
        # create per-run log directory logs/<model_id>/<model_id>_N
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

        if SummaryWriter is not None:
            try:
                self.writer = SummaryWriter(log_dir=run_dir)
            except Exception:
                self.writer = None
        else:
            self.writer = None
            print("[warning] TensorBoard SummaryWriter not available — no logs will be written")

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
                    # raw episode length (divided by two for comparability with other agents)
                    self.writer.add_scalar("random/episode_length", l / 2, i)
                except Exception:
                    pass
        avg = sum(lengths) / len(lengths) if lengths else 0.0
        print(f"avg episodes {avg}")
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass
        return avg

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Random agent runner")
    p.add_argument("--model-id", default="random")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--window", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agent = RandomAgent(model_id=args.model_id, size=args.size, episodes=args.episodes, window=args.window)
    avg = agent.run()
    print(f"Final average episode length: {avg}")