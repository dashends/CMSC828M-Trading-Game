# CMSC828M Trading Game

Done:
1. Implement the game environment. We can use OpenAI Gym following this trading game example: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

TODO:
0. change 13 cards to a parameter to constructor
0. randomize turn sequence, change at end of each day
0. add more sequencies per round
1. test the environment (testing starter code in main.py)
	a. obs correct?
	b. actions correct?
	c. reward correct?
	d. result correct?
2. implement baseline agents in baseline_agents.py


3. training:  starts with 2 players

4. implement self-play
5. self-play training
6. playing against the agent





Action: offer sell, offer buy.  With an amount + a price

observation_space: public pile + own hand + money+ transection history + contract each player has + sequence of players
none of the Stable Baselines can handle Dict/Tuple spaces. Concatenate them into Box space.


reward: expected profit * timestep
=  (expected value of public pile * amount of contract + cureent balance – initial balance) * timestep
times timestep to incentive late game profits more than in the beginning

= 9+5 + 

91-9-5- 3- 4 =


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