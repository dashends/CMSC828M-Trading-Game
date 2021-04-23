"""
This script loads models listed in trained_models and evaluates them against EVAgent
"""

import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.env_checker import check_env

# we need to use the same settings as the env used in training. Otherwise the agent may be confused.
NUM_PLAYERS = 2
SEQ_PER_DAY = 3
CARDS_PER_SUIT = 10
SUIT_COUNT = 1
BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
EVAL_EPISODES = int(1e3)

hand_count = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - hand_count*NUM_PLAYERS

# add 1 baseline agent
agents = [baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = hand_count, public_cards_count = PUBLIC_CARDS_COUNT)]
env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents, 
	seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, public_cards_count = PUBLIC_CARDS_COUNT, 
	random_seq = True)


# load the trained model
trained_models = ["model_10cards_2seq_2player_1suit", "model_checkpoints/rl_model_100000_steps", "model_checkpoints/rl_model_200000_steps", "model_checkpoints/rl_model_300000_steps",
	"model_checkpoints/rl_model_400000_steps", "model_checkpoints/rl_model_500000_steps", "model_checkpoints/rl_model_600000_steps",
	"model_checkpoints/rl_model_700000_steps", "model_checkpoints/rl_model_800000_steps", "model_checkpoints/rl_model_900000_steps",
	"model_final"]
trained_models = ["model_checkpoints/rl_model_300000_steps", "model_checkpoints/rl_model_600000_steps",
	"model_checkpoints/rl_model_900000_steps", "model_checkpoints/rl_model_1200000_steps",
	"model_checkpoints/rl_model_1500000_steps", "model_checkpoints/rl_model_1800000_steps",
	"model_checkpoints/rl_model_2100000_steps", "model_checkpoints/rl_model_2400000_steps",
	"model_checkpoints/rl_model_2700000_steps", "model_checkpoints/rl_model_3000000_steps", 
	"model_checkpoints/rl_model_3300000_steps",
	"model_checkpoints/rl_model_3600000_steps", "model_checkpoints/rl_model_3900000_steps",
	"model_checkpoints/rl_model_4200000_steps", "model_checkpoints/rl_model_4500000_steps",
	"model_checkpoints/rl_model_4800000_steps", "model_checkpoints/rl_model_5100000_steps",
	"model_checkpoints/rl_model_5400000_steps","model_checkpoints/rl_model_5700000_steps",
	"model_checkpoints/rl_model_6000000_steps"]
	
for model_path in trained_models:
	model = PPO2.load(model_path)


	# Evaluate the agent
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES)
	print(model_path, " mean_reward: ", mean_reward, " std_reward: ", std_reward)

	
# some results:
"""
model trained against baseline  		 mean_reward:  24.5027431640625  std_reward:  16.8649353503965

self-play
model_checkpoints/rl_model_100000_steps  mean_reward:  7.075248046875  std_reward:  11.385631573191768
model_checkpoints/rl_model_200000_steps  mean_reward:  15.82374609375  std_reward:  18.806204943864273
model_checkpoints/rl_model_300000_steps  mean_reward:  8.2827509765625  std_reward:  12.43023233553445
model_checkpoints/rl_model_400000_steps  mean_reward:  1.2677548828125  std_reward:  5.576380655727703
model_checkpoints/rl_model_500000_steps  mean_reward:  2.187755859375  std_reward:  8.68180080544172
model_checkpoints/rl_model_600000_steps  mean_reward:  7.8022470703125  std_reward:  13.991453528735686
model_checkpoints/rl_model_700000_steps  mean_reward:  3.75601171875  std_reward:  9.897077008967933
model_checkpoints/rl_model_800000_steps  mean_reward:  12.616248046875  std_reward:  18.243143446531946
model_checkpoints/rl_model_900000_steps  mean_reward:  8.347998046875  std_reward:  14.80434875653121
model_final  							 mean_reward:  5.2892490234375  std_reward:  10.666854558644227
"""

"""
after fixing EVAgent bug

model trained against baseline			 mean_reward:  -0.1835068359375  std_reward:  7.564737798144282

self-play
model_checkpoints/rl_model_100000_steps  mean_reward:  -1.1140048828125  std_reward:  6.670322420880464
model_checkpoints/rl_model_200000_steps  mean_reward:  -3.655505859375  std_reward:  10.788276451579106
model_checkpoints/rl_model_300000_steps  mean_reward:  -0.1190009765625  std_reward:  2.548194233270012
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.3045  std_reward:  2.170490670332402
model_checkpoints/rl_model_500000_steps  mean_reward:  -0.16899609375  std_reward:  1.976602122105493
model_checkpoints/rl_model_600000_steps  mean_reward:  -1.04649609375  std_reward:  6.884482897110345
model_checkpoints/rl_model_700000_steps  mean_reward:  -0.9744931640625  std_reward:  4.998286991872872
model_checkpoints/rl_model_800000_steps  mean_reward:  -5.7400107421875  std_reward:  15.073361501028499
model_checkpoints/rl_model_900000_steps  mean_reward:  -4.4385087890625  std_reward:  13.077519331783087
model_final  							 mean_reward:  0.0015  std_reward:  0.8007794640223985
"""

"""
model_checkpoints/rl_model_400000_steps  mean_reward:  0.33549609375  std_reward:  6.269014671245232
model_checkpoints/rl_model_800000_steps  mean_reward:  -0.4534921875  std_reward:  3.9475349712482477
model_final2  							 mean_reward:  0.2565  std_reward:  3.7827847791553704
"""

"""
reward without multipling day number
model_checkpoints/rl_model_300000_steps  mean_reward:  -0.018  std_reward:  0.23168081491569395
model_checkpoints/rl_model_600000_steps  mean_reward:  -0.3225  std_reward:  1.8363669976341874
model_checkpoints/rl_model_900000_steps  mean_reward:  -0.3815009765625  std_reward:  2.4844952749925957
model_checkpoints/rl_model_1200000_steps  mean_reward:  -0.955001953125  std_reward:  5.468488620280933
model_checkpoints/rl_model_1500000_steps  mean_reward:  -0.327  std_reward:  1.997015523224594
model_checkpoints/rl_model_1800000_steps  mean_reward:  -0.056501953125  std_reward:  0.9841512205705021
model_checkpoints/rl_model_2100000_steps  mean_reward:  -0.8805  std_reward:  5.922245785386853
model_checkpoints/rl_model_2400000_steps  mean_reward:  -0.953501953125  std_reward:  6.2800520525284576
model_checkpoints/rl_model_2700000_steps  mean_reward:  -0.06950390625  std_reward:  5.987359676143568
model_checkpoints/rl_model_3000000_steps  mean_reward:  -5.40148828125  std_reward:  15.342710524537749
model_checkpoints/rl_model_3300000_steps  mean_reward:  -0.015  std_reward:  0.3908644266238615
model_checkpoints/rl_model_3600000_steps  mean_reward:  -0.4010009765625  std_reward:  3.4945944550476233
model_checkpoints/rl_model_3900000_steps  mean_reward:  -2.356998046875  std_reward:  7.428502413615852
model_checkpoints/rl_model_4200000_steps  mean_reward:  -0.003  std_reward:  0.09482088377567466
model_checkpoints/rl_model_4500000_steps  mean_reward:  -3.6409951171875  std_reward:  10.320758876130165
model_checkpoints/rl_model_4800000_steps  mean_reward:  -0.6465  std_reward:  3.6197082410050676
model_checkpoints/rl_model_5100000_steps  mean_reward:  -0.2109990234375  std_reward:  2.3804310687618733
model_checkpoints/rl_model_5400000_steps  mean_reward:  0.075498046875  std_reward:  1.1621131958560482
model_checkpoints/rl_model_5700000_steps  mean_reward:  0.2284921875  std_reward:  5.291457990890346
model_checkpoints/rl_model_6000000_steps  mean_reward:  -0.478001953125  std_reward:  4.675327578947858
"""

# Playing test rounds
model = PPO2.load("model_checkpoints/rl_model_200000_steps")
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
#print("EVAgent betting range ", agents[0].val)

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
