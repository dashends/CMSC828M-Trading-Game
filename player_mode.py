"""
This script loads a models and let you play against the model
"""

import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.env_checker import check_env
from os import listdir
from os.path import isfile, join

# we need to use the same settings as the env used in training. Otherwise the agent may be confused.
NUM_PLAYERS = 2
SEQ_PER_DAY = 2
CARDS_PER_SUIT = 10
SUIT_COUNT = 1
BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
EVAL_EPISODES = int(1e3)
TRANSACTION_HISTORY_SIZE = 4

HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

# no  player agent
agents = []
agents.append(baseline_agents.PlayerAgent())


env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
	seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
	random_seq = True, self_play = False, obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, eval=True)


#
# Playing against it
model = PPO2.load("saved_agent")
print("Playing test rounds:")
obs = env.reset()

print("public pile", env.public_pile)
print("hands", env.hands)
print("Groud Truth Value of Future's", env.public_pile.sum())


#print("cards per suit ", agents[0].cards_per_suit)
#print("player's cards ", agents[0].player_hand_count)
#print("EVAgent betting range ", agents[0].val)

env.render()
'''
Which obs passed in for baseline_agent
'''
while(True):
	print('Taking actions for the current sequence......')
	action, _states = model.predict(obs)
	print("other agent's action: ", action)
	obs, reward, done, info = env.step(action)
	print("total net worth: ", env.balance + env.contract * env.public_pile.sum())
	env.render()
	if done:
		break
