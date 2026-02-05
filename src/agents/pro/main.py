"""PPO agent and Gymnasium wrapper for the Battleship environment.

This module provides a small Gymnasium-compatible wrapper around the
project's `Battleship` environment and a `PPOAgent` that uses
stable-baselines3 to train a policy.

Requirements:
- gymnasium
- stable-baselines3
- numpy

The wrapper exposes a flattened observation (vector) so we can use
`MlpPolicy` from stable-baselines3 by default.
"""

from __future__ import annotations

import argparse
import os
import random
import importlib.util
from typing import Optional

import numpy as np

try:
	import gymnasium as gym
except Exception:  # pragma: no cover
	raise ImportError("Please install 'gymnasium' to use the PPO agent")

try:
	from stable_baselines3 import PPO
	from stable_baselines3.common.vec_env import DummyVecEnv
	from stable_baselines3.common.callbacks import BaseCallback, CallbackList
	from stable_baselines3.common.monitor import Monitor
	from stable_baselines3.common.logger import configure
except Exception:  # pragma: no cover
	raise ImportError("Please install 'stable-baselines3' to use the PPO agent")

try:
	from torch.utils.tensorboard import SummaryWriter
except Exception:
	SummaryWriter = None


def _load_battleship_class():
	"""Try importing the project's Battleship class with fallbacks.

	The repository sometimes doesn't expose packages on PYTHONPATH when
	the module is executed directly. Load by module name first, then
	fall back to loading by file path relative to this file.
	"""
	try:
		# common case when running from project root where src is on sys.path
		from envs.battleship import Battleship  # type: ignore
		return Battleship
	except Exception:
		# try loading from relative path: ../../envs/battleship.py
		base = os.path.dirname(__file__)
		candidate = os.path.abspath(os.path.join(base, "..", "..", "envs", "battleship.py"))
		if os.path.exists(candidate):
			spec = importlib.util.spec_from_file_location("battleship_local", candidate)
			module = importlib.util.module_from_spec(spec)
			assert spec and spec.loader
			spec.loader.exec_module(module)
			return getattr(module, "Battleship")
		raise

# Load Battleship class and provide a Gym wrapper
Battleship = _load_battleship_class()


class BattleshipGym(gym.Env):
	"""Gymnasium wrapper around the project's `Battleship` game.

	Single-agent environment where the learning agent plays as player 1
	and the opponent plays random valid moves. Observations are flattened
	vectors so SB3's `MlpPolicy` can be used.
	"""

	metadata = {"render.modes": ["human"]}

	def __init__(self, size: int = 6, seed: Optional[int] = None, step_penalty: float = 0.0, writer=None, agent_prefix: str = "pro"):
		super().__init__()
		self.size = size
		self.bs = Battleship(size)
		self.step_penalty = float(step_penalty)
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# bookkeeping for external callbacks
		self.writer = writer
		self.agent_prefix = agent_prefix or "pro"
		self._ep_steps = 0
		self._episodes = 0
		self._ep_lengths = []

		obs_shape = (4 * size * size,)
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
		self.action_space = gym.spaces.Discrete(size * size)
		self.state = None
		self.current_player = 1

	def reset(self, *, seed: Optional[int] = None, options=None):
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
		self.state = self.bs.restart(1)
		self.current_player = 1
		self._ep_steps = 0
		obs = self.bs.get_encoded_state(self.state).astype(np.float32).flatten()
		return obs, {}

	def step(self, action: int):
		assert self.state is not None, "Call reset() before step()"
		# agent move
		self.state = self.bs.step(self.state, int(action), 1)
		self._ep_steps += 1
		done = False
		reward = 0.0
		reward += float(getattr(self, "step_penalty", 0.0))
		if getattr(self.bs, "repeat", False):
			reward += 1.0
		else:
			reward -= 0.1

		winner, terminal = self.bs.terminated(self.state, action)
		if terminal:
			done = True
			if self.bs.check_win(self.state, action, 1):
				reward += 10.0
			else:
				reward -= 10.0
			# bookkeeping for external logger callbacks
			self._ep_lengths.append(self._ep_steps)
			self._episodes += 1
			obs = self.bs.get_encoded_state(self.state).astype(np.float32).flatten()
			return obs, reward, done, False, {}

		# opponent moves if no repeat
		if not getattr(self.bs, "repeat", False):
			opp_turn = True
			while opp_turn:
				valid = self.bs.get_valid_moves(self.state, -1)
				valid_idx = np.where(np.asarray(valid).flatten() == 1)[0]
				if len(valid_idx) == 0:
					break
				a = int(np.random.choice(valid_idx))
				self.state = self.bs.step(self.state, a, -1)
				if getattr(self.bs, "repeat", False):
					reward -= 0.5
					if self.bs.check_win(self.state, a, -1):
						done = True
						reward -= 10.0
						opp_turn = False
						break
					opp_turn = True
				else:
					opp_turn = False

		obs = self.bs.get_encoded_state(self.state).astype(np.float32).flatten()
		return obs, reward, done, False, {}

	def render(self, mode: str = "human"):
		print("Render not implemented in detail. Use Battleship debug methods.")

	def render(self, mode: str = "human"):
		# Minimal textual render: show hit and ship layers sizes
		print("Render not implemented in detail. Use debug in Battleship class.")


class PPOAgent:
	"""Simple wrapper for stable-baselines3 PPO on the BattleshipGym."""

	def __init__(self, size: int = 6, log_dir: Optional[str] = "logs/pro", step_penalty: float = 0.0):
		self.size = size
		self.log_dir = log_dir
		self.step_penalty = float(step_penalty)
		self.model: Optional[PPO] = None
		# create per-run timestamped directory under provided log_dir (e.g. logs/pro/pro_YYYY_MM_DD...)
		base_log = self.log_dir or os.path.join("logs", "pro")
		try:
			os.makedirs(base_log, exist_ok=True)
		except Exception:
			pass
		try:
			from datetime import datetime
			ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		except Exception:
			ts = "run"
		run_name = f"pro_{ts}"
		self.run_dir = os.path.join(base_log, run_name)
		try:
			os.makedirs(self.run_dir, exist_ok=True)
		except Exception:
			pass
		# Do not create a separate SummaryWriter here to avoid duplicate TB writers.
		# Logging of episode statistics is performed via SB3 logger in a callback.
		self.writer = None
		# create matching model directory for this run
		self.model_dir = os.path.join("models", "pro", run_name)
		try:
			os.makedirs(self.model_dir, exist_ok=True)
		except Exception:
			pass

	def _make_vec_env(self):
		# inject step_penalty into the env constructor and pass the writer
		# capture the Env class in a local variable so the lambda resolves it reliably
		env_cls = BattleshipGym
		return DummyVecEnv([
			lambda: Monitor(env_cls(self.size, step_penalty=self.step_penalty, writer=self.writer, agent_prefix="pro"))
		])

	def train(self, total_timesteps: int = 10000, show_progress: bool = True, **ppo_kwargs):
		env = self._make_vec_env()
		# SB3 tensorboard logging: write into the per-run directory created at init
		# configure SB3 logger to write TensorBoard events directly into the run directory
		self.model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None, **ppo_kwargs)
		try:
			logger = configure(self.run_dir, ["tensorboard"])  # write TB events into run_dir
			self.model.set_logger(logger)
		except Exception:
			# fallback: let SB3 handle tensorboard logging under the provided target
			pass
		callbacks = None
		if show_progress:
			class TqdmCallback(BaseCallback):
				def __init__(self, total_timesteps):
					super().__init__()
					self.total = int(total_timesteps)
					self.pbar = None
					self._last_num_timesteps = 0

				def _on_training_start(self) -> None:
					from tqdm.auto import tqdm
					self.pbar = tqdm(total=self.total, desc="PPO Training", unit="timestep")

				def _on_step(self) -> bool:
					# update by delta timesteps since last call
					delta = int(self.num_timesteps - self._last_num_timesteps)
					if delta > 0 and self.pbar is not None:
						self.pbar.update(delta)
						self._last_num_timesteps = int(self.num_timesteps)
						return True

				def _on_training_end(self) -> None:
					if self.pbar is not None:
						self.pbar.close()

		callbacks = [TqdmCallback(total_timesteps)]

		# Callback to read env episode bookkeeping and write into SB3 logger
		class SB3EpisodeCallback(BaseCallback):
			def __init__(self):
				super().__init__()
				self._last_eps = 0

			def _find_battleship_env(self):
				for e in getattr(self.training_env, 'envs', []):
					cand = e
					for _ in range(4):
						if hasattr(cand, 'env'):
							cand = getattr(cand, 'env')
						elif hasattr(cand, 'unwrapped'):
							cand = getattr(cand, 'unwrapped')
						else:
							break
					if hasattr(cand, '_ep_lengths'):
						return cand
				return None

			def _on_step(self) -> bool:
				bs_env = self._find_battleship_env()
				if bs_env is None:
					return True
				total_eps = getattr(bs_env, '_episodes', 0)
				if total_eps > self._last_eps:
					recent = getattr(bs_env, '_ep_lengths', [])[-100:]
					mean_len = float(sum(recent)) / len(recent) if recent else 0.0
					try:
						self.logger.record('ppo/episode_length', recent[-1] if recent else 0.0)
						self.logger.dump(self.num_timesteps)
					except Exception:
						pass
					self._last_eps = total_eps
				return True

		callbacks = CallbackList(callbacks + [SB3EpisodeCallback()])

		self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
		# ensure writer flush so TensorBoard sees latest scalars
		try:
			if getattr(self, "writer", None) is not None:
				self.writer.flush()
		except Exception:
			pass
		return self.model

	def save(self, path: str):
		if self.model is None:
			raise RuntimeError("No model to save. Train or load a model first.")
		# save into model_dir created for this run
		try:
			os.makedirs(self.model_dir, exist_ok=True)
		except Exception:
			pass
		save_path = os.path.join(self.model_dir, "main")
		self.model.save(save_path)

	def load(self, path: str):
		env = self._make_vec_env()
		self.model = PPO.load(path, env=env)
		return self.model

	def act(self, obs: np.ndarray, deterministic: bool = True):
		if self.model is None:
			raise RuntimeError("No model loaded. Train or load a model first.")
		action, _ = self.model.predict(obs, deterministic=deterministic)
		return int(action)

	def evaluate(self, episodes: int = 50):
		env = BattleshipGym(self.size)
		wins = 0
		total_reward = 0.0
		for _ in range(episodes):
			obs, _ = env.reset()
			done = False
			ep_reward = 0.0
			while not done:
				action = self.act(obs, deterministic=True)
				obs, r, done, truncated, info = env.step(action)
				ep_reward += r
			total_reward += ep_reward
			# crude win detection: reward > 0 indicates wins more likely
			if ep_reward > 0:
				wins += 1
		return {"win_rate": wins / episodes, "avg_reward": total_reward / episodes}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--size", type=int, default=5)
	parser.add_argument("--timesteps", type=int, default=20000)
	parser.add_argument("--save", type=str, default=None)
	parser.add_argument("--logdir", type=str, default="logs/pro")
	parser.add_argument("--step-penalty", type=float, default=0.0, help="per-step reward penalty (negative encourages shorter episodes)")
	parser.add_argument("--eval_episodes", type=int, default=50)
	args = parser.parse_args()

	agent = PPOAgent(size=args.size, log_dir=args.logdir, step_penalty=args.step_penalty)
	print("Training PPO on Battleship...")
	agent.train(total_timesteps=args.timesteps)
	agent.save(None)
	print(f"Saved model to {agent.model_dir}")
	print("Evaluating model...")
	stats = agent.evaluate(episodes=args.eval_episodes)
	print(stats)


if __name__ == "__main__":
	main()

