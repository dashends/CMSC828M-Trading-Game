import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.env_checker import check_env

# we need to use the same settings as the env used in training. Otherwise the agent may be confused.
NUM_PLAYERS = 2
SEQ_PER_DAY = 2
CARDS_PER_SUIT = 10
SUIT_COUNT = 1
BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100

hand_count = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - hand_count*NUM_PLAYERS

# add 1 baseline agent
agents = [baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = hand_count, public_cards_count = PUBLIC_CARDS_COUNT)]
env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents, seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, public_cards_count = PUBLIC_CARDS_COUNT, random_seq = True)


# load the trained model
model = PPO2.load("model_1")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean_reward: ", mean_reward)
print("std_reward: ", std_reward)

# Playing test rounds
print("Playing test rounds:")
obs = env.reset()

print("obs space", env.observation_space)
print("action space", env.action_space)
print("sample action", env.action_space.sample())
print("public pile", env.public_pile)
print("hands", env.hands)
print("Groud Truth Value of Future's", env.public_pile.sum())


print("suit sum ", agents[0].SUIT_SUM)

#print("cards per suit ", agents[0].cards_per_suit)
#print("player's cards ", agents[0].player_hand_count)
print("betting range ", agents[0].val)

env.render()
print('obs=', obs)
'''
Which obs passed in for baseline_agent
'''
while(True):
	print('Taking actions for the current sequence......')
	action, _states = model.predict(obs)
	print("action: ", action)
	obs, reward, done, info = env.step(action)
	print('reward=', reward, 'done=', done)
	print("total net worth: ", env.balance + env.contract * env.public_pile.sum())
	env.render()
	print('obs=', obs)
	if done:
		break
