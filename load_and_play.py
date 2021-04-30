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
SEQ_PER_DAY = 2
CARDS_PER_SUIT = 10
SUIT_COUNT = 1
BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
EVAL_EPISODES = int(1e3)
TRANSACTION_HISTORY_SIZE = 4

HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

# add 2 baseline agent
agents = []
agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))
#agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
	seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
	random_seq = True, self_play = False, obs_transaction_history_size=TRANSACTION_HISTORY_SIZE)


# load the trained model
trained_models = ["model_checkpoints/rl_model_400000_steps", "model_checkpoints/rl_model_800000_steps",
	"model_checkpoints/rl_model_1200000_steps", "model_checkpoints/rl_model_1600000_steps",
	"model_checkpoints/rl_model_2000000_steps", "model_checkpoints/rl_model_2400000_steps",
	"model_checkpoints/rl_model_2800000_steps", "model_checkpoints/rl_model_3200000_steps",
	"model_checkpoints/rl_model_3600000_steps", "model_checkpoints/rl_model_4000000_steps", 
	"model_checkpoints/rl_model_4400000_steps",
	"model_checkpoints/rl_model_4800000_steps", "model_checkpoints/rl_model_5200000_steps",
	"model_checkpoints/rl_model_5600000_steps",
	"model_checkpoints/rl_model_6000000_steps", "model_checkpoints/rl_model_6400000_steps",
	"model_checkpoints/rl_model_6800000_steps", "model_checkpoints/rl_model_7200000_steps",
	"model_checkpoints/rl_model_7600000_steps", "model_checkpoints/rl_model_8000000_steps",
	"model_checkpoints/rl_model_8400000_steps","model_checkpoints/rl_model_8800000_steps",
	"model_checkpoints/rl_model_9200000_steps", "model_checkpoints/rl_model_9600000_steps", "model_final"]
	
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

"""
After extending observation space 

Trained against the agent (1 sequence of transaction history)
model_checkpoints/rl_model_400000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_800000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.046673828125  std_reward:  5.151487836224526
model_checkpoints/rl_model_1600000_steps  mean_reward:  0.7586787109375  std_reward:  5.486966680990354
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.02201171875  std_reward:  5.7460785819262545
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.2288515625  std_reward:  5.354851361985972
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.7818359375  std_reward:  5.583552612306864
model_checkpoints/rl_model_3200000_steps  mean_reward:  0.786669921875  std_reward:  5.982272021911061
model_checkpoints/rl_model_3600000_steps  mean_reward:  0.569673828125  std_reward:  6.428014357344575
model_checkpoints/rl_model_4000000_steps  mean_reward:  0.6436640625  std_reward:  5.918118228685374
model_checkpoints/rl_model_4400000_steps  mean_reward:  0.391001953125  std_reward:  5.344479176614119
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.6689970703125  std_reward:  5.386773980929091
model_checkpoints/rl_model_5200000_steps  mean_reward:  0.968021484375  std_reward:  5.980174184409271
model_checkpoints/rl_model_5600000_steps  mean_reward:  0.7806650390625  std_reward:  5.82259192058807
model_checkpoints/rl_model_6000000_steps  mean_reward:  1.071009765625  std_reward:  5.29454618068977
model_checkpoints/rl_model_6400000_steps  mean_reward:  1.304998046875  std_reward:  5.951794663405608
model_checkpoints/rl_model_6800000_steps  mean_reward:  1.2956884765625  std_reward:  6.300181602159656
model_checkpoints/rl_model_7200000_steps  mean_reward:  1.2161650390625  std_reward:  5.172339422056622
model_checkpoints/rl_model_7600000_steps  mean_reward:  1.63550390625  std_reward:  5.478624545143309
model_checkpoints/rl_model_8000000_steps  mean_reward:  1.7593564453125  std_reward:  6.189203120986271
model_checkpoints/rl_model_8400000_steps  mean_reward:  1.3054990234375  std_reward:  5.803298658891032
model_checkpoints/rl_model_8800000_steps  mean_reward:  1.477001953125  std_reward:  5.139578044498456
model_checkpoints/rl_model_9200000_steps  mean_reward:  4.061322265625  std_reward:  5.500769415491867
model_checkpoints/rl_model_9600000_steps  mean_reward:  4.144357421875  std_reward:  5.0166000479739505


Trained against the agent (4 sequence of transaction history)
model_checkpoints/rl_model_400000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_800000_steps  mean_reward:  1.4834951171875  std_reward:  2.8514683433056756
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.5771728515625  std_reward:  2.953405263428362
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.4753349609375  std_reward:  2.77246064665549
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.7636689453125  std_reward:  2.9797155998822955
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.763001953125  std_reward:  3.029126158029976
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.8564990234375  std_reward:  2.9244631005493247
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.861005859375  std_reward:  3.0274767449837987
model_checkpoints/rl_model_3600000_steps  mean_reward:  2.767162109375  std_reward:  2.8233649531142744
model_checkpoints/rl_model_4000000_steps  mean_reward:  2.700990234375  std_reward:  2.7587223981512694
model_checkpoints/rl_model_4400000_steps  mean_reward:  2.78067578125  std_reward:  2.8477947852586802
model_checkpoints/rl_model_4800000_steps  mean_reward:  2.8456728515625  std_reward:  2.7701445384202583
model_checkpoints/rl_model_5200000_steps  mean_reward:  2.7924912109375  std_reward:  2.766952178250764
model_checkpoints/rl_model_5600000_steps  mean_reward:  2.7128291015625  std_reward:  2.70261801721013
model_checkpoints/rl_model_6000000_steps  mean_reward:  2.8018203125  std_reward:  2.7795629707226066
model_checkpoints/rl_model_6400000_steps  mean_reward:  2.7558349609375  std_reward:  2.7410286747454755
model_checkpoints/rl_model_6800000_steps  mean_reward:  2.7118310546875  std_reward:  2.7424176856956763
model_checkpoints/rl_model_7200000_steps  mean_reward:  2.7996640625  std_reward:  2.8286492962114695
model_checkpoints/rl_model_7600000_steps  mean_reward:  2.8924931640625  std_reward:  2.8450188245353334
model_checkpoints/rl_model_8000000_steps  mean_reward:  2.8210029296875  std_reward:  2.8322221957367373
model_checkpoints/rl_model_8400000_steps  mean_reward:  2.729853515625  std_reward:  2.7172413186989206
model_checkpoints/rl_model_8800000_steps  mean_reward:  2.835318359375  std_reward:  2.661514191429318
model_checkpoints/rl_model_9200000_steps  mean_reward:  2.8795  std_reward:  2.748002449087304
model_checkpoints/rl_model_9600000_steps  mean_reward:  2.787326171875  std_reward:  2.8221458122133813
model_final  							  mean_reward:  2.7699921875  std_reward:  2.8420802866919166




Self-play:  (1 sequence of transaction history)
model_checkpoints/rl_model_400000_steps  mean_reward:  -10.719529296875  std_reward:  20.03518042312496
model_checkpoints/rl_model_800000_steps  mean_reward:  -3.237669921875  std_reward:  10.373702373389518
model_checkpoints/rl_model_1200000_steps  mean_reward:  -0.031  std_reward:  7.028261833467808
model_checkpoints/rl_model_1600000_steps  mean_reward:  -1.36066796875  std_reward:  9.451557628337042
model_checkpoints/rl_model_2000000_steps  mean_reward:  -1.071181640625  std_reward:  8.556192365158275
model_checkpoints/rl_model_2400000_steps  mean_reward:  -0.6490224609375  std_reward:  7.835970457101601
model_checkpoints/rl_model_2800000_steps  mean_reward:  -0.11950390625  std_reward:  7.382711580076602
model_checkpoints/rl_model_3200000_steps  mean_reward:  -0.994337890625  std_reward:  6.868585778700112
model_checkpoints/rl_model_3600000_steps  mean_reward:  0.25149609375  std_reward:  3.637157125794526
model_checkpoints/rl_model_4000000_steps  mean_reward:  0.2648291015625  std_reward:  4.668861873034483
model_checkpoints/rl_model_4400000_steps  mean_reward:  -0.6118349609375  std_reward:  3.99508298613912
model_checkpoints/rl_model_4800000_steps  mean_reward:  0.3271669921875  std_reward:  3.060395784313317
model_checkpoints/rl_model_5200000_steps  mean_reward:  -0.8183564453125  std_reward:  3.9292486180350807
model_checkpoints/rl_model_5600000_steps  mean_reward:  -0.2648486328125  std_reward:  4.812138065844983
model_checkpoints/rl_model_6000000_steps  mean_reward:  -0.28766796875  std_reward:  3.4851087934153386
model_checkpoints/rl_model_6400000_steps  mean_reward:  -0.740837890625  std_reward:  3.5862353913252636
model_checkpoints/rl_model_6800000_steps  mean_reward:  -0.4959951171875  std_reward:  2.981099524516976
model_checkpoints/rl_model_7200000_steps  mean_reward:  -0.5801630859375  std_reward:  2.9383083434849713
model_checkpoints/rl_model_7600000_steps  mean_reward:  -0.2913291015625  std_reward:  4.457757342955978
model_checkpoints/rl_model_8000000_steps  mean_reward:  -0.501181640625  std_reward:  3.920442668616256
model_checkpoints/rl_model_8400000_steps  mean_reward:  -0.1756640625  std_reward:  2.595466472617155

Self-play with dynamic sampling and evaluation  (1 sequence of transaction history)
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.259  std_reward:  1.8583376980516755
model_checkpoints/rl_model_800000_steps  mean_reward:  -6.236009765625  std_reward:  15.86963226371016
model_checkpoints/rl_model_1200000_steps  mean_reward:  -1.5373388671875  std_reward:  7.233184256504164
model_checkpoints/rl_model_1600000_steps  mean_reward:  -0.019  std_reward:  0.40575731663150566
model_checkpoints/rl_model_2000000_steps  mean_reward:  -2.8273466796875  std_reward:  8.896241428541616
model_checkpoints/rl_model_2400000_steps  mean_reward:  -6.5668388671875  std_reward:  14.829721042119676
model_checkpoints/rl_model_2800000_steps  mean_reward:  -0.001  std_reward:  0.03160696125855822
model_checkpoints/rl_model_3200000_steps  mean_reward:  -5.7981416015625  std_reward:  14.08650012961464
model_checkpoints/rl_model_3600000_steps  mean_reward:  -6.9938330078125  std_reward:  14.344033494853822
model_checkpoints/rl_model_4000000_steps  mean_reward:  0.206833984375  std_reward:  1.4086148271795045
model_checkpoints/rl_model_4400000_steps  mean_reward:  -0.3629990234375  std_reward:  1.8435809562851535
model_checkpoints/rl_model_4800000_steps  mean_reward:  0.2579833984375  std_reward:  5.677139855016549
model_checkpoints/rl_model_5200000_steps  mean_reward:  -1.5686767578125  std_reward:  6.210958843640776
model_checkpoints/rl_model_5600000_steps  mean_reward:  -0.6384970703125  std_reward:  6.481134358017889
model_checkpoints/rl_model_6000000_steps  mean_reward:  -0.5020068359375  std_reward:  7.031841412707224
model_checkpoints/rl_model_6400000_steps  mean_reward:  0.215666015625  std_reward:  1.8745401940268085
model_checkpoints/rl_model_6800000_steps  mean_reward:  0.57298828125  std_reward:  4.007670461226491
model_checkpoints/rl_model_7200000_steps  mean_reward:  0.625001953125  std_reward:  3.724128138257788
model_checkpoints/rl_model_7600000_steps  mean_reward:  -0.2956669921875  std_reward:  7.542196820204915
model_checkpoints/rl_model_8000000_steps  mean_reward:  -0.4241748046875  std_reward:  7.218626430162291
model_checkpoints/rl_model_8400000_steps  mean_reward:  0.2346630859375  std_reward:  1.4814530000853061
model_checkpoints/rl_model_8800000_steps  mean_reward:  0.147666015625  std_reward:  3.117393954124543
model_checkpoints/rl_model_9200000_steps  mean_reward:  0.001662109375  std_reward:  2.3982728670649114
model_checkpoints/rl_model_9600000_steps  mean_reward:  -1.6861689453125  std_reward:  8.673128540233568
model_final  											mean_reward:  -4.2816591796875  std_reward:  12.529649336545484

Self-play with dynamic sampling and evaluation  (4 sequence of transaction history)
model_checkpoints/rl_model_400000_steps  mean_reward:  -6.910849609375  std_reward:  10.161558508899331
model_checkpoints/rl_model_800000_steps  mean_reward:  -3.3596669921875  std_reward:  8.400261776773313
model_checkpoints/rl_model_1200000_steps  mean_reward:  -4.7780009765625  std_reward:  8.235351820512081
model_checkpoints/rl_model_1600000_steps  mean_reward:  -1.2815126953125  std_reward:  3.3784208458804446
model_checkpoints/rl_model_2000000_steps  mean_reward:  -1.32816796875  std_reward:  4.375028935152989
model_checkpoints/rl_model_2400000_steps  mean_reward:  -0.6926591796875  std_reward:  3.155274873918684
model_checkpoints/rl_model_2800000_steps  mean_reward:  -0.5136552734375  std_reward:  2.728271952812285
model_checkpoints/rl_model_3200000_steps  mean_reward:  -1.17384375  std_reward:  3.8656040111201997
model_checkpoints/rl_model_3600000_steps  mean_reward:  -2.70017578125  std_reward:  7.150000204713192
model_checkpoints/rl_model_4000000_steps  mean_reward:  -2.826337890625  std_reward:  5.607164800092301
model_checkpoints/rl_model_4400000_steps  mean_reward:  -0.09033984375  std_reward:  1.5254168837818671
model_checkpoints/rl_model_4800000_steps  mean_reward:  -0.631671875  std_reward:  2.751857169611597
model_checkpoints/rl_model_5200000_steps  mean_reward:  -0.2225  std_reward:  1.0008714952480164
"""

"""
3-player game  (1 sequence of transaction history)

Training against two EVAgents gives 0. buy at very low and sell at very high

Self-play
model_checkpoints/rl_model_400000_steps  mean_reward:  -3.783703125  std_reward:  11.518490941838234
model_checkpoints/rl_model_800000_steps  mean_reward:  -9.218158203125  std_reward:  17.585053665646623
model_checkpoints/rl_model_1200000_steps  mean_reward:  -1.3338505859375  std_reward:  5.032715236974004
model_checkpoints/rl_model_1600000_steps  mean_reward:  0.023  std_reward:  0.43642983399396523
model_checkpoints/rl_model_2000000_steps  mean_reward:  -1.7788505859375  std_reward:  6.893397339900614
model_checkpoints/rl_model_2400000_steps  mean_reward:  -4.0525517578125  std_reward:  11.46620892738626
model_checkpoints/rl_model_2800000_steps  mean_reward:  -0.732994140625  std_reward:  6.320214309786262
model_checkpoints/rl_model_3200000_steps  mean_reward:  -1.1886015625  std_reward:  6.143008714167037
model_checkpoints/rl_model_3600000_steps  mean_reward:  -0.1315498046875  std_reward:  2.405867831936909
model_checkpoints/rl_model_4000000_steps  mean_reward:  -0.8916513671875  std_reward:  4.775970532353448
"""

# Playing test rounds
model = PPO2.load(trained_models[-1])
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
