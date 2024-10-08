# Alphazero implementation for the game battleship

This program solved the game of battleship using alphazero.

## Battleship

The game battleship is a board game with two players whereas each player has two maps. 
At first each player places a set of discrete ships on a discrete ship map.
After which the battle phase begins. During the phase each player guesses the position of the opponents ships.
Hits on the others shipmap are being stored on the respective hitmaps.  

## Alphazero

Alphazero is a model-based deep reinforcement learning algorithm.<br/>
The most capable agent achives victory in about 29 moves in the mean on a 9x9 battleship game while playing against a human which needs around 42 moves in the mean.

## Getting started

Training
```
python3 alphazero.py
```

Playing against alphazero in the browser
```
python3 gameserver.py
```
