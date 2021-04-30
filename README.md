# CMSC828M Trading Game

Usage Command: 
	1. python train.py						#This script trains the agent and saves resulting model as SAVE_NAME.zip. It also saves copies of past models to model_checkpoints folder.
	2. python load_and_play.py				#This script loads models listed in trained_models and evaluates them against EVAgent
	
To get best training speed, it is recommanded to run the script in front (bring focus to the window), because some OS (such as Windows 10) 
may automatically slow down processes that run in the background.

Done:
1. Implement the game environment
	a. init()
	b. reset()
	c. step()
	d. render()
	e. next_observation
	f. take_action
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





TODO:
1. test the environment (testing starter code in main.py)
	a. obs correct?
	b. actions correct?
	c. reward correct?
	d. result correct?
3. add obs spaces (Amir)
3. change action space to number of contracts to buy (Da)
3. train and observe (Da)
4. self-play training
3. plot training results (mean rewards vs num of time steps) (DQN vs PPO2) (MLP policy vs. RNN policy etc.), quality scores of past selves
3. add baseline agent to opponent list

4. randomization to force exploration. e.g. cost multiplier of contracts. Initial states of the but/sell offer markets. 
	Sufficiently diverse training games are necessary to ensure robustness to the wide variety of strategies and situations that arise in games against human opponents.
	
5. playing against the agent
6. how to let the agent scale to different setups of the game env? For example, play against 3 players with 4 suits and then 4 players with 2 suits?
7. for continuous spaces, normalize observation/action space if possible (A good practice is to rescale your actions to lie in [-1, 1])




Action: offer sell, offer buy.  With an amount + a price

observation_space: public pile + own hand + money+ transection history + contract each player has + sequence of players
none of the Stable Baselines can handle Dict/Tuple spaces. Concatenate them into Box space.


reward: 
1. expected profit * timestep + panelty if no action
=  (expected value of public pile * amount of contract + cureent balance – initial balance) * timestep + panelty if no action
times timestep to incentive late game profits more than in the beginning
2. give reward based on ground truth. no need to times timestep because profit is determined for each action. No need to incentive late game profits more than in the beginning

Start with 1 suit, 1 contract, 1 sequence per round, 2 agents


Training: play against baseline model; self-play 

Training against baseline models is easier to start with. But the performance might not be very good

Self-play: 
1. To avoid “strategy collapse”, the agent trains 80% of its games against itself and the other 20% against its past selves.
2. To force exploration in strategy space, during training (and only during training) we may randomize the properties of the units.



User testing: we can play against the agent (see https://github.com/openai/gym/blob/master/gym/utils/play.py#L26




package versions:
Python 3.6.13
numpy                1.19.5
gym                  0.18.0
stable-baselines     2.10.1
tensorflow           1.15.0
(warning: stable baseline works with tensorflow 1.x)