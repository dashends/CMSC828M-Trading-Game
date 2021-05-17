# CMSC828M Trading Game

Usage Command: 
	1. python player_mode.py				#This script loads a models and let you play against the model
	2. python train.py						#This script trains the agent and saves resulting model as SAVE_NAME.zip. It also saves copies of past models to model_checkpoints folder.
	3. python load_and_play.py				#This script loads models listed in trained_models and evaluates them against EVAgent
	4. python experiment.py					#This script runs the experiments mentioned in the paper
	5. python hyperparam_tuning/tuning		#This script runs hyper-parameter tuning using optuna
	

To get best training speed, it is recommanded to run the script in front (bring focus to the window), because some OS (such as Windows 10) may automatically slow down processes that run in the background.




package versions:
Python 				 3.6.13
numpy                1.19.5
gym                  0.18.0
stable-baselines     2.10.1
tensorflow           1.15.0
(warning: stable baseline only works with tensorflow 1.x)


What is done so far:
1. Implement the game environment
	a. init()
	b. reset()
	c. step()
	d. render()
	e. next_observation
	f. take_action
1. test the environment (testing starter code in main.py)
2. [1,1,100,10] should post two offers. also disallow self trading
3. implement more suits
4. change card numbers to a parameter to constructor
5. change sequences per day to a parameter to constructor. 		
	self.sequence_counter represents the current sequence number of that day; self.day represents day number. 
	e.g. day 3 sequence 2 is self.sequence_counter = 2
6. randomize turn sequence. change at end of each day
7. implement baseline agents in baseline_agents.py 	(Amir)
8. implement self-play 
9. implement dynamic sampling and evaluation for self play
9. training:  starts with 2 players
10. extend obs spaces (Amir)
11. add baseline agent to opponent list
12. remove bad models from model bank
13.  add penalty for rediculously high/low price for some margins
14. custom policy network
15. plot training results 
	mean rewards vs num of time steps:
	2 player, 10 cards, 4 sequences, 20% EVAgent, 10 updates
	(1) relu vs. tanh, different networks arch
	(2) transaction history length		
	(3) % of EVAgent
	(4) model bank update frequency
	(5) dynamic sampling and evaluation (on and off)
	(6) larger games (more cards) (more players 30 cards)
	(7) MLP policy vs. RNN policy



Future directions:
1. force exploration




