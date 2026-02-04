# Alphazero implementation for the game battleship

This program solved the game of battleship using alphazero and provides a web interface for playing.

![screenshot](images/Game.png)

## Getting started

To make sure pyenv is active in the shell:

```bash
export PYENV_ROOT="$HOME/.pyenv" && export PATH="$PYENV_ROOT/bin:$PATH" && eval "$(pyenv init -)"
```

Create a pyenv:

```bash
pyenv virtualenv 3.12.2 alphazero
```

Activate the pyenv:

```bash
pyenv activate alphazero
```

Install libs: 

```bash
poetry install
```

Starting a training run with Alphazero:

```bash
poetry run python3 src/main.py
```

Starting a training run with the random Agent:

```bash
poetry run python3 src/main.py --agent random --episodes 1024 --size 5
```

Tensorboard for live stats:

```bash
poetry run tensorboard --logdir logs --port 6006
```

## Eval

Comparing different agents performance:

```bash
poetry run python3 eval/plot_avg_episodes.py --logdir logs --smooth 0.5
```

## Battleship

The game battleship is a board game with two players whereas each player has two maps. 
At first each player places a set of discrete ships on a discrete ship map.
After which the battle phase begins. During the phase each player guesses the position of the opponents ships.
Hits on the others shipmap are being stored on the respective hitmaps.  

## Alphazero

Alphazero is a model-based deep reinforcement learning algorithm.<br/>
The most capable agent achives victory in about 29 moves in the mean on a 9x9 battleship game with ```ships = [5, 4, 3, 2]``` while playing against a human which needs around 42 moves in the mean. 

## Getting started

Start a training process with ```python3 alphazero.py```.

To play against a model first change into the django server directory ```cd gameserver_django```. Then run ```python3 manage.py runserver``` to start the django server locally.
