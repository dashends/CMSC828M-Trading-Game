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

# add 2 baseline agent
agents = []
agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))
#agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
	seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
	random_seq = True, self_play = False, obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, eval=True)


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
#trained_models = [join("model_checkpoints/", f) for f in listdir("model_checkpoints/") if isfile(join("model_checkpoints/", f))]
	
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


Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
model_checkpoints/rl_model_400000_steps  mean_reward:  0.1133232421875  std_reward:  2.5206279005588725
model_checkpoints/rl_model_800000_steps  mean_reward:  1.164337890625  std_reward:  2.2739943196125934
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.5081767578125  std_reward:  2.866311885312222
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.4661650390625  std_reward:  3.1634863724839346
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.405333984375  std_reward:  2.9470859893243935
model_checkpoints/rl_model_2400000_steps  mean_reward:  2.4611689453125  std_reward:  3.0784348618797064
model_checkpoints/rl_model_2800000_steps  mean_reward:  2.4496796875  std_reward:  2.9654965428493156
model_checkpoints/rl_model_3200000_steps  mean_reward:  2.7131591796875  std_reward:  2.9943266963187867
model_checkpoints/rl_model_3600000_steps  mean_reward:  2.5476748046875  std_reward:  2.775348933549834
model_checkpoints/rl_model_4000000_steps  mean_reward:  2.712853515625  std_reward:  2.9122123311878303
model_checkpoints/rl_model_4400000_steps  mean_reward:  2.6143466796875  std_reward:  2.8161872406935307
model_checkpoints/rl_model_4800000_steps  mean_reward:  2.7191650390625  std_reward:  2.876511519838676
model_checkpoints/rl_model_5200000_steps  mean_reward:  2.6551826171875  std_reward:  2.7886083433987627
model_checkpoints/rl_model_5600000_steps  mean_reward:  2.425158203125  std_reward:  2.8977934399204535
model_checkpoints/rl_model_6000000_steps  mean_reward:  2.7671787109375  std_reward:  2.8596325653837518
model_checkpoints/rl_model_6400000_steps  mean_reward:  2.63917578125  std_reward:  2.850962400359063
model_checkpoints/rl_model_6800000_steps  mean_reward:  2.651  std_reward:  2.7337032413929943
model_checkpoints/rl_model_7200000_steps  mean_reward:  2.7154990234375  std_reward:  2.9152669271347835
model_checkpoints/rl_model_7600000_steps  mean_reward:  2.5345087890625  std_reward:  2.684110378492743
model_checkpoints/rl_model_8000000_steps  mean_reward:  2.58766796875  std_reward:  2.8177097656123014
model_checkpoints/rl_model_8400000_steps  mean_reward:  2.798015625  std_reward:  2.8791266067666683
model_checkpoints/rl_model_8800000_steps  mean_reward:  2.4700146484375  std_reward:  2.7318037163984807
model_checkpoints/rl_model_9200000_steps  mean_reward:  2.66283203125  std_reward:  2.637804152393363
model_checkpoints/rl_model_9600000_steps  mean_reward:  2.653994140625  std_reward:  3.0683338630378123
model_final  mean_reward:  2.8691669921875  std_reward:  2.9172073293739973
	
Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified, 
	update model bank every 1280 timesteps
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.006  std_reward:  0.0772269383052313
model_checkpoints/rl_model_800000_steps  mean_reward:  0.0456669921875  std_reward:  0.4474935354596992
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.2183271484375  std_reward:  3.117883852552721
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.42899609375  std_reward:  3.2560826054247474
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.29566796875  std_reward:  3.0920142207318335
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.3085029296875  std_reward:  2.7603903525418336
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.4160009765625  std_reward:  2.8777861207442097
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.972669921875  std_reward:  3.089674425627628
model_checkpoints/rl_model_3600000_steps  mean_reward:  1.9956708984375  std_reward:  3.0449438246368445
model_checkpoints/rl_model_4000000_steps  mean_reward:  2.017990234375  std_reward:  2.902482279562598
model_checkpoints/rl_model_4400000_steps  mean_reward:  2.013341796875  std_reward:  2.9244806004182604
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.901826171875  std_reward:  2.8270007644229933


Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 50% EV Agent, reward not modified
model_checkpoints/rl_model_400000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_800000_steps  mean_reward:  1.4771572265625  std_reward:  2.9151401590311616
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.6878369140625  std_reward:  2.998270332479165
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.730837890625  std_reward:  2.9119016569916454
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.8274990234375  std_reward:  3.0090709684209402
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.763162109375  std_reward:  3.0177346654121076
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.5589931640625  std_reward:  2.6473412242461953
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.634166015625  std_reward:  2.834857232100744
model_checkpoints/rl_model_3600000_steps  mean_reward:  1.63917578125  std_reward:  2.975021506439178
model_checkpoints/rl_model_4000000_steps  mean_reward:  1.761494140625  std_reward:  2.801508725077332
model_checkpoints/rl_model_4400000_steps  mean_reward:  1.7326640625  std_reward:  2.8188573891249997
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.81433203125  std_reward:  2.955287528374559
model_checkpoints/rl_model_5200000_steps  mean_reward:  1.688  std_reward:  2.826298807365712
model_checkpoints/rl_model_5600000_steps  mean_reward:  1.6888330078125  std_reward:  2.770768855250091
model_checkpoints/rl_model_6000000_steps  mean_reward:  1.6994951171875  std_reward:  2.9049605695689555
model_checkpoints/rl_model_6400000_steps  mean_reward:  1.6594990234375  std_reward:  2.819800108134037
model_checkpoints/rl_model_6800000_steps  mean_reward:  1.822173828125  std_reward:  2.8988942815782472
model_checkpoints/rl_model_7200000_steps  mean_reward:  1.8093271484375  std_reward:  2.8025512133569888
model_checkpoints/rl_model_7600000_steps  mean_reward:  1.5563349609375  std_reward:  2.6245938233943256
model_checkpoints/rl_model_8000000_steps  mean_reward:  1.716837890625  std_reward:  2.8895624964507385
model_checkpoints/rl_model_8400000_steps  mean_reward:  1.666669921875  std_reward:  2.8215404326695257
model_checkpoints/rl_model_8800000_steps  mean_reward:  1.71916796875  std_reward:  2.912454536445955
model_checkpoints/rl_model_9200000_steps  mean_reward:  1.590998046875  std_reward:  2.7913241939833138
model_checkpoints/rl_model_9600000_steps  mean_reward:  1.6871728515625  std_reward:  2.853103643761518
model_final  mean_reward:  1.6748408203125  std_reward:  2.8085507465629944


Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 10% EV Agent, reward not modified
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.08633203125  std_reward:  0.8515272460269948
model_checkpoints/rl_model_800000_steps  mean_reward:  -0.0548330078125  std_reward:  0.689979795076488
model_checkpoints/rl_model_1200000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_1600000_steps  mean_reward:  -0.113998046875  std_reward:  1.0880582552751512
model_checkpoints/rl_model_2000000_steps  mean_reward:  0.028  std_reward:  0.4440996113635233
model_checkpoints/rl_model_2400000_steps  mean_reward:  0.067333984375  std_reward:  0.8380605452955198
model_checkpoints/rl_model_2800000_steps  mean_reward:  0.061330078125  std_reward:  0.8102160179853385
model_checkpoints/rl_model_3200000_steps  mean_reward:  0.1673310546875  std_reward:  2.2961287344850736
model_checkpoints/rl_model_3600000_steps  mean_reward:  0.5268310546875  std_reward:  2.1659173772766276
model_checkpoints/rl_model_4000000_steps  mean_reward:  0.4616640625  std_reward:  2.0255305385166076
model_checkpoints/rl_model_4400000_steps  mean_reward:  0.7944951171875  std_reward:  2.5188333506139187


Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy128*128*128
model_checkpoints/rl_model_400000_steps  mean_reward:  0.0  std_reward:  0.0
model_checkpoints/rl_model_800000_steps  mean_reward:  0.0806630859375  std_reward:  0.6010414686629985
model_checkpoints/rl_model_1200000_steps  mean_reward:  0.086666015625  std_reward:  0.707136602645396
model_checkpoints/rl_model_1600000_steps  mean_reward:  0.26999609375  std_reward:  1.6039557223261305
model_checkpoints/rl_model_2000000_steps  mean_reward:  0.65433203125  std_reward:  1.5810617649094605
model_checkpoints/rl_model_2400000_steps  mean_reward:  0.973990234375  std_reward:  2.9042836188141856
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.2249970703125  std_reward:  2.7560084769361857
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.259333984375  std_reward:  2.8934330639615755
model_checkpoints/rl_model_3600000_steps  mean_reward:  1.26198828125  std_reward:  2.6915190742351123
model_checkpoints/rl_model_4000000_steps  mean_reward:  1.249671875  std_reward:  2.5691084386792618
model_checkpoints/rl_model_4400000_steps  mean_reward:  1.480498046875  std_reward:  3.2663820415513047
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.5249912109375  std_reward:  2.742308298906338
model_checkpoints/rl_model_5200000_steps  mean_reward:  1.418169921875  std_reward:  2.683098086951714
model_checkpoints/rl_model_5600000_steps  mean_reward:  1.4130068359375  std_reward:  2.6444554582947912
model_checkpoints/rl_model_6000000_steps  mean_reward:  1.2388291015625  std_reward:  2.847346201031052
model_checkpoints/rl_model_6400000_steps  mean_reward:  1.4740078125  std_reward:  3.0290269260084988
model_checkpoints/rl_model_6800000_steps  mean_reward:  1.6713359375  std_reward:  3.30344187524607
model_checkpoints/rl_model_7200000_steps  mean_reward:  1.5386552734375  std_reward:  2.9542123473401234
model_checkpoints/rl_model_7600000_steps  mean_reward:  1.63233203125  std_reward:  3.253563380234771
model_checkpoints/rl_model_8000000_steps  mean_reward:  1.2249892578125  std_reward:  3.4503878322488872
model_checkpoints/rl_model_8400000_steps  mean_reward:  1.66433203125  std_reward:  3.1680050445357533
model_checkpoints/rl_model_8400000_steps  mean_reward:  1.473998046875  std_reward:  2.968296222599647
model_checkpoints/rl_model_8800000_steps  mean_reward:  1.8293291015625  std_reward:  3.1091583453601697
model_checkpoints/rl_model_9200000_steps  mean_reward:  1.83016015625  std_reward:  2.6751906312672364
model_checkpoints/rl_model_9600000_steps  mean_reward:  1.9621650390625  std_reward:  3.2319025331814926
model_final  mean_reward:  								1.4281630859375  std_reward:  3.219838588335453


Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy ReLU
model_checkpoints/rl_model_400000_steps  mean_reward:  0.87766015625  std_reward:  1.9473764327526546
model_checkpoints/rl_model_800000_steps  mean_reward:  1.364998046875  std_reward:  2.8102827257480967
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.96866015625  std_reward:  3.521939784883168
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.959837890625  std_reward:  3.471578988315738
model_checkpoints/rl_model_2000000_steps  mean_reward:  2.264998046875  std_reward:  3.6122802210595766
model_checkpoints/rl_model_2400000_steps  mean_reward:  2.3559970703125  std_reward:  3.434580050110897
model_checkpoints/rl_model_2800000_steps  mean_reward:  2.9400078125  std_reward:  3.8376261838086903
model_checkpoints/rl_model_3200000_steps  mean_reward:  3.0251728515625  std_reward:  3.4613611800212807
model_checkpoints/rl_model_3600000_steps  mean_reward:  2.7394833984375  std_reward:  3.4177484097001667
model_checkpoints/rl_model_4000000_steps  mean_reward:  2.867482421875  std_reward:  3.5798751040995827
model_checkpoints/rl_model_4400000_steps  mean_reward:  3.007498046875  std_reward:  3.9143417204366617
model_checkpoints/rl_model_4800000_steps  mean_reward:  2.828998046875  std_reward:  3.305721742186843
model_checkpoints/rl_model_5200000_steps  mean_reward:  3.002  std_reward:  4.108381065432653
model_checkpoints/rl_model_5600000_steps  mean_reward:  3.1736533203125  std_reward:  3.3400773828300916
model_checkpoints/rl_model_6000000_steps  mean_reward:  3.1613193359375  std_reward:  3.501694017604422
model_checkpoints/rl_model_6400000_steps  mean_reward:  3.139494140625  std_reward:  3.348073980600462
model_checkpoints/rl_model_6800000_steps  mean_reward:  3.193828125  std_reward:  3.5440821792490755
model_checkpoints/rl_model_7200000_steps  mean_reward:  3.1293291015625  std_reward:  3.3429032463987474
model_checkpoints/rl_model_7600000_steps  mean_reward:  3.2573203125  std_reward:  3.515589638491865
model_checkpoints/rl_model_8000000_steps  mean_reward:  3.0729921875  std_reward:  3.4273793531178933
model_checkpoints/rl_model_8400000_steps  mean_reward:  3.102173828125  std_reward:  3.2894639601355986
model_checkpoints/rl_model_8800000_steps  mean_reward:  3.1081748046875  std_reward:  3.2080832855584864
model_checkpoints/rl_model_9200000_steps  mean_reward:  3.00149609375  std_reward:  3.154759030181732
model_checkpoints/rl_model_9600000_steps  mean_reward:  3.08732421875  std_reward:  3.2996713338232464
model_final  							  mean_reward:  3.2319990234375  std_reward:  3.79073395518368

Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy ReLU n_step=1024
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.3429951171875  std_reward:  2.165286113314749
model_checkpoints/rl_model_800000_steps  mean_reward:  0.356662109375  std_reward:  1.4371519256840612
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.1691630859375  std_reward:  2.982496168172666
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.501828125  std_reward:  3.3782661415018786
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.38282421875  std_reward:  3.4053265828454307
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.5423291015625  std_reward:  3.7031020780055126
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.313828125  std_reward:  3.3010926152144884
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.455169921875  std_reward:  3.4680472653142407
model_checkpoints/rl_model_3600000_steps  mean_reward:  1.927  std_reward:  3.6798199976165966
model_checkpoints/rl_model_4000000_steps  mean_reward:  1.2988369140625  std_reward:  3.1418422039831304
model_checkpoints/rl_model_4400000_steps  mean_reward:  1.3189970703125  std_reward:  3.4058738342028634
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.542  std_reward:  3.7462103857869007
model_checkpoints/rl_model_5200000_steps  mean_reward:  1.4531572265625  std_reward:  3.399105244024132
model_checkpoints/rl_model_5600000_steps  mean_reward:  1.995166015625  std_reward:  3.3538491701271202
model_checkpoints/rl_model_6000000_steps  mean_reward:  2.2488203125  std_reward:  3.6021200320288376
model_checkpoints/rl_model_6400000_steps  mean_reward:  2.1563193359375  std_reward:  3.6036381275172498
model_checkpoints/rl_model_6800000_steps  mean_reward:  2.5460029296875  std_reward:  4.092166145743942
model_checkpoints/rl_model_7200000_steps  mean_reward:  2.5051708984375  std_reward:  3.802082505286488
model_checkpoints/rl_model_7600000_steps  mean_reward:  2.4086669921875  std_reward:  3.655016441905095
model_checkpoints/rl_model_7600000_steps  mean_reward:  2.581173828125  std_reward:  3.8472387814743376
model_checkpoints/rl_model_8000000_steps  mean_reward:  2.424171875  std_reward:  3.6954016861388324
model_checkpoints/rl_model_8400000_steps  mean_reward:  2.7983330078125  std_reward:  4.10865758050359
model_checkpoints/rl_model_8800000_steps  mean_reward:  2.4463291015625  std_reward:  3.684079099013987
model_checkpoints/rl_model_9200000_steps  mean_reward:  2.6315  std_reward:  3.871941304802343
model_checkpoints/rl_model_9600000_steps  mean_reward:  2.466001953125  std_reward:  3.7313049899452015

Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy ReLU SELF_COPY_FREQ = 1, EveryNTimesteps(n_steps=128*num_cpu
model_checkpoints/rl_model_400000_steps  mean_reward:  0.0136669921875  std_reward:  0.22640681312127697
model_checkpoints/rl_model_800000_steps  mean_reward:  0.4719990234375  std_reward:  1.7597125137415597
model_checkpoints/rl_model_1200000_steps  mean_reward:  1.4283271484375  std_reward:  2.8975532329614624
model_checkpoints/rl_model_1600000_steps  mean_reward:  1.5518349609375  std_reward:  3.2833585993579044
model_checkpoints/rl_model_2000000_steps  mean_reward:  1.5061591796875  std_reward:  3.286261016077214
model_checkpoints/rl_model_2400000_steps  mean_reward:  1.5478310546875  std_reward:  3.043940178502492
model_checkpoints/rl_model_2800000_steps  mean_reward:  1.4650078125  std_reward:  3.426169120974623
model_checkpoints/rl_model_3200000_steps  mean_reward:  1.597166015625  std_reward:  3.1779353334432736
model_checkpoints/rl_model_3600000_steps  mean_reward:  1.5906591796875  std_reward:  2.9365751746848887
model_checkpoints/rl_model_4000000_steps  mean_reward:  1.4930029296875  std_reward:  3.240271162732739
model_checkpoints/rl_model_4400000_steps  mean_reward:  1.8358359375  std_reward:  3.1655920515665827
model_checkpoints/rl_model_4800000_steps  mean_reward:  1.82599609375  std_reward:  3.0084659750131606
model_checkpoints/rl_model_5200000_steps  mean_reward:  1.9155009765625  std_reward:  3.395921158533506
model_checkpoints/rl_model_5600000_steps  mean_reward:  2.2869912109375  std_reward:  3.4142045095640365
model_checkpoints/rl_model_6000000_steps  mean_reward:  2.8740009765625  std_reward:  3.625231754980626
model_checkpoints/rl_model_6400000_steps  mean_reward:  2.63249609375  std_reward:  3.856784859761176
model_checkpoints/rl_model_6800000_steps  mean_reward:  2.8388310546875  std_reward:  3.7352843761358407
model_checkpoints/rl_model_7200000_steps  mean_reward:  2.7466728515625  std_reward:  3.5745863737031063
model_checkpoints/rl_model_7600000_steps  mean_reward:  2.61850390625  std_reward:  3.143164456768349
model_checkpoints/rl_model_8000000_steps  mean_reward:  2.8461767578125  std_reward:  3.1277611098811375
model_checkpoints/rl_model_8400000_steps  mean_reward:  2.92599609375  std_reward:  3.2876294177439656
model_checkpoints/rl_model_8800000_steps  mean_reward:  2.95366796875  std_reward:  3.6084698205874877
model_checkpoints/rl_model_9200000_steps  mean_reward:  3.020669921875  std_reward:  3.52168571585561
model_checkpoints/rl_model_9600000_steps  mean_reward:  2.93101171875  std_reward:  3.344447097880035
model_final  							  mean_reward:  3.0866640625  std_reward:  3.466085863277032



Best model:
Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy ReLU
model_checkpoints/rl_model_400000_steps.zip  mean_reward:  -0.004  std_reward:  0.13410443691392168
model_checkpoints/rl_model_800000_steps.zip  mean_reward:  0.2218291015625  std_reward:  1.6697191379100664
model_checkpoints/rl_model_1200000_steps.zip  mean_reward:  1.3270087890625  std_reward:  2.932519707276288
model_checkpoints/rl_model_1600000_steps.zip  mean_reward:  1.4968330078125  std_reward:  3.066866304242758
model_checkpoints/rl_model_2000000_steps.zip  mean_reward:  1.4376591796875  std_reward:  3.225527038746946
model_checkpoints/rl_model_2400000_steps.zip  mean_reward:  1.7093388671875  std_reward:  3.0171366509380406
model_checkpoints/rl_model_2800000_steps.zip  mean_reward:  1.72465625  std_reward:  3.533463781684803
model_checkpoints/rl_model_3200000_steps.zip  mean_reward:  1.3901650390625  std_reward:  3.7191738492544757
model_checkpoints/rl_model_3600000_steps.zip  mean_reward:  1.3458349609375  std_reward:  3.52121544481777
model_checkpoints/rl_model_4000000_steps.zip  mean_reward:  2.457994140625  std_reward:  4.332617527556192
model_checkpoints/rl_model_4400000_steps.zip  mean_reward:  2.6440068359375  std_reward:  4.123749788651543
model_checkpoints/rl_model_4800000_steps.zip  mean_reward:  2.5005  std_reward:  3.887499192394896
model_checkpoints/rl_model_5200000_steps.zip  mean_reward:  2.52533984375  std_reward:  3.7071745427657428
model_checkpoints/rl_model_5600000_steps.zip  mean_reward:  2.6576650390625  std_reward:  3.8608183261411835
model_checkpoints/rl_model_6000000_steps.zip  mean_reward:  2.4965  std_reward:  3.725809745200502
model_checkpoints/rl_model_6400000_steps.zip  mean_reward:  2.640666015625  std_reward:  3.415150023307592
model_checkpoints/rl_model_6800000_steps.zip  mean_reward:  2.7138388671875  std_reward:  3.664801754834221
model_checkpoints/rl_model_7200000_steps.zip  mean_reward:  2.66915234375  std_reward:  3.857393081510416
model_checkpoints/rl_model_7600000_steps.zip  mean_reward:  2.8941455078125  std_reward:  3.8298181564567186
model_checkpoints/rl_model_8000000_steps.zip  mean_reward:  2.9340009765625  std_reward:  3.973023859453966
model_checkpoints/rl_model_8400000_steps.zip  mean_reward:  2.6991640625  std_reward:  3.920230364753063
model_checkpoints/rl_model_8800000_steps.zip  mean_reward:  2.7464951171875  std_reward:  3.6123052028523754
model_checkpoints/rl_model_9200000_steps.zip  mean_reward:  2.7259970703125  std_reward:  3.963649133520537
model_checkpoints/rl_model_9600000_steps.zip  mean_reward:  2.97450390625  std_reward:  4.16406147194523
model_checkpoints/rl_model_10000000_steps.zip  mean_reward:  2.5163232421875  std_reward:  3.6845250007793493
model_checkpoints/rl_model_10400000_steps.zip  mean_reward:  2.80349609375  std_reward:  4.049015183934491
model_checkpoints/rl_model_10800000_steps.zip  mean_reward:  2.2954873046875  std_reward:  3.654476377466289
model_checkpoints/rl_model_11200000_steps.zip  mean_reward:  2.68649609375  std_reward:  3.7852215220604113
model_checkpoints/rl_model_11600000_steps.zip  mean_reward:  2.644494140625  std_reward:  3.694616773933107
model_checkpoints/rl_model_12000000_steps.zip  mean_reward:  2.63217578125  std_reward:  3.975195783063663
model_checkpoints/rl_model_12400000_steps.zip  mean_reward:  3.039837890625  std_reward:  3.785826863732178205
model_checkpoints/rl_model_12800000_steps.zip  mean_reward:  2.7546650390625  std_reward:  3.68529806146335
model_checkpoints/rl_model_13200000_steps.zip  mean_reward:  3.0189921875  std_reward:  3.7647197103459162
model_checkpoints/rl_model_13600000_steps.zip  mean_reward:  2.9219853515625  std_reward:  3.5864368559743016
model_checkpoints/rl_model_14000000_steps.zip  mean_reward:  2.902837890625  std_reward:  3.7062081634018016
model_checkpoints/rl_model_14400000_steps.zip  mean_reward:  3.0045107421875  std_reward:  3.7895822726914714
model_checkpoints/rl_model_14800000_steps.zip  mean_reward:  3.00465625  std_reward:  3.7979199898333857
model_checkpoints/rl_model_15200000_steps.zip  mean_reward:  2.84948828125  std_reward:  3.730318405253301
model_checkpoints/rl_model_15600000_steps.zip  mean_reward:  3.069998046875  std_reward:  3.6685177994138094
model_checkpoints/rl_model_16000000_steps.zip  mean_reward:  3.405671875  std_reward:  4.014984565878119
model_checkpoints/rl_model_16400000_steps.zip  mean_reward:  2.97166796875  std_reward:  3.8347038947331216
model_checkpoints/rl_model_16800000_steps.zip  mean_reward:  3.153501953125  std_reward:  3.714749557684555
model_checkpoints/rl_model_17200000_steps.zip  mean_reward:  3.0393369140625  std_reward:  3.9388715461273223
model_checkpoints/rl_model_17600000_steps.zip  mean_reward:  3.1145078125  std_reward:  4.153887641463456
model_checkpoints/rl_model_18000000_steps.zip  mean_reward:  3.1631748046875  std_reward:  3.8162262782218304
model_checkpoints/rl_model_18400000_steps.zip  mean_reward:  2.978826171875  std_reward:  4.006751742418233
model_checkpoints/rl_model_18800000_steps.zip  mean_reward:  3.2323330078125  std_reward:  4.037706936085268
model_checkpoints/rl_model_19200000_steps.zip  mean_reward:  3.1881640625  std_reward:  4.269387870437462
model_checkpoints/rl_model_19600000_steps.zip  mean_reward:  3.2251728515625  std_reward:  3.5762257711463783
model_checkpoints/rl_model_20000000_steps.zip  mean_reward:  3.420322265625  std_reward:  3.6873165459855657
model_checkpoints/rl_model_20400000_steps.zip  mean_reward:  3.08917578125  std_reward:  4.1821510828987325
model_checkpoints/rl_model_20800000_steps.zip  mean_reward:  2.599490234375  std_reward:  4.3323082281634475
model_checkpoints/rl_model_21200000_steps.zip  mean_reward:  3.337837890625  std_reward:  4.157426856571888
model_checkpoints/rl_model_21600000_steps.zip  mean_reward:  3.2436611328125  std_reward:  4.216427687220295
model_checkpoints/rl_model_22000000_steps.zip  mean_reward:  3.258171875  std_reward:  3.7630414149397504
model_checkpoints/rl_model_22400000_steps.zip  mean_reward:  3.40833984375  std_reward:  3.949789099121118
model_checkpoints/rl_model_22800000_steps.zip  mean_reward:  3.3700048828125  std_reward:  4.3019490457652445
model_checkpoints/rl_model_23200000_steps.zip  mean_reward:  3.3446787109375  std_reward:  3.771137262947097
model_checkpoints/rl_model_23600000_steps.zip  mean_reward:  3.3705146484375  std_reward:  3.747782573893905
model_checkpoints/rl_model_24000000_steps.zip  mean_reward:  3.2678330078125  std_reward:  4.120451447520917
model_checkpoints/rl_model_24400000_steps.zip  mean_reward:  3.124845703125  std_reward:  3.939446839195279
model_checkpoints/rl_model_24800000_steps.zip  mean_reward:  3.570837890625  std_reward:  4.206533102742982
model_checkpoints/rl_model_25200000_steps.zip  mean_reward:  3.7143515625  std_reward:  4.303827362914588
model_checkpoints/rl_model_25600000_steps.zip  mean_reward:  3.23399609375  std_reward:  3.6492500654945648
model_checkpoints/rl_model_26000000_steps.zip  mean_reward:  3.027337890625  std_reward:  4.607329746435695
model_checkpoints/rl_model_26400000_steps.zip  mean_reward:  3.2803310546875  std_reward:  3.9046783710348905
model_checkpoints/rl_model_26800000_steps.zip  mean_reward:  3.6641787109375  std_reward:  4.1107447738840435
model_checkpoints/rl_model_27200000_steps.zip  mean_reward:  3.4703310546875  std_reward:  4.072988052751381
model_checkpoints/rl_model_27600000_steps.zip  mean_reward:  3.41066015625  std_reward:  3.633217427940781
model_checkpoints/rl_model_28000000_steps.zip  mean_reward:  3.353337890625  std_reward:  4.145270264155633
model_checkpoints/rl_model_28400000_steps.zip  mean_reward:  3.384173828125  std_reward:  4.2486002386043245
model_checkpoints/rl_model_28800000_steps.zip  mean_reward:  3.514828125  std_reward:  4.068737161518708
model_checkpoints/rl_model_29200000_steps.zip  mean_reward:  3.2884892578125  std_reward:  3.757661086927187
model_checkpoints/rl_model_29600000_steps.zip  mean_reward:  3.4540078125  std_reward:  4.126479809220658
model_checkpoints/rl_model_30000000_steps.zip  mean_reward:  3.359328125  std_reward:  3.987286871317042
model_checkpoints/rl_model_30400000_steps.zip  mean_reward:  3.2343330078125  std_reward:  3.9063551215287684
model_checkpoints/rl_model_30800000_steps.zip  mean_reward:  3.4808359375  std_reward:  4.140989630151186
model_checkpoints/rl_model_31200000_steps.zip  mean_reward:  3.3940078125  std_reward:  3.861423808996803
model_checkpoints/rl_model_31600000_steps.zip  mean_reward:  3.2911708984375  std_reward:  4.02127363588954
model_checkpoints/rl_model_32000000_steps.zip  mean_reward:  3.15915625  std_reward:  4.05766528102891
model_checkpoints/rl_model_32400000_steps.zip  mean_reward:  3.464486328125  std_reward:  3.984437256096324
model_checkpoints/rl_model_32800000_steps.zip  mean_reward:  3.3095029296875  std_reward:  4.222580447409021
model_checkpoints/rl_model_33200000_steps.zip  mean_reward:  3.54433984375  std_reward:  4.058653328896419
model_checkpoints/rl_model_33600000_steps.zip  mean_reward:  3.4583525390625  std_reward:  3.8271744318647025
model_checkpoints/rl_model_34000000_steps.zip  mean_reward:  3.1138310546875  std_reward:  3.9258079001095925
model_checkpoints/rl_model_34400000_steps.zip  mean_reward:  3.0443291015625  std_reward:  3.7651825659400595
model_checkpoints/rl_model_34800000_steps.zip  mean_reward:  3.583833984375  std_reward:  4.031723529332098
model_checkpoints/rl_model_35200000_steps.zip  mean_reward:  3.158822265625  std_reward:  3.775349912537029
model_checkpoints/rl_model_35600000_steps.zip  mean_reward:  3.3323330078125  std_reward:  3.845664429262567
model_checkpoints/rl_model_36000000_steps.zip  mean_reward:  3.4229951171875  std_reward:  4.21872395660172
model_checkpoints/rl_model_36400000_steps.zip  mean_reward:  3.256333984375  std_reward:  4.07267414179727
model_checkpoints/rl_model_36800000_steps.zip  mean_reward:  3.53498828125  std_reward:  3.9911947553004166
model_checkpoints/rl_model_37200000_steps.zip  mean_reward:  3.316171875  std_reward:  4.0884089844976215
model_checkpoints/rl_model_37600000_steps.zip  mean_reward:  3.42383984375  std_reward:  3.945847320970975
model_checkpoints/rl_model_38000000_steps.zip  mean_reward:  3.13183984375  std_reward:  3.863362345059341
model_checkpoints/rl_model_38400000_steps.zip  mean_reward:  3.2501572265625  std_reward:  4.176621735772218
model_checkpoints/rl_model_38800000_steps.zip  mean_reward:  3.4059970703125  std_reward:  3.964430486168875
model_checkpoints/rl_model_39200000_steps.zip  mean_reward:  3.5446650390625  std_reward:  3.9723802685841463
model_checkpoints/rl_model_39600000_steps.zip  mean_reward:  3.575662109375  std_reward:  4.2126401507232885
model_checkpoints/rl_model_40000000_steps.zip  mean_reward:  3.558005859375  std_reward:  4.107587877704064
model_checkpoints/rl_model_40400000_steps.zip  mean_reward:  3.313826171875  std_reward:  3.914020305089175
model_checkpoints/rl_model_40800000_steps.zip  mean_reward:  3.6066650390625  std_reward:  3.8379208441944774
model_checkpoints/rl_model_41200000_steps.zip  mean_reward:  3.4910029296875  std_reward:  3.9640454047822398
model_checkpoints/rl_model_41600000_steps.zip  mean_reward:  3.457841796875  std_reward:  3.9353586099777265
model_checkpoints/rl_model_42000000_steps.zip  mean_reward:  3.473509765625  std_reward:  4.0410873094470885
model_checkpoints/rl_model_42400000_steps.zip  mean_reward:  3.442173828125  std_reward:  3.9778558795818135
model_checkpoints/rl_model_42800000_steps.zip  mean_reward:  3.32716015625  std_reward:  4.082563294453946
model_checkpoints/rl_model_43200000_steps.zip  mean_reward:  3.629505859375  std_reward:  4.070757738360119
model_checkpoints/rl_model_43600000_steps.zip  mean_reward:  3.3356708984375  std_reward:  4.2046425408769394
model_checkpoints/rl_model_44000000_steps.zip  mean_reward:  3.6585068359375  std_reward:  4.039855544165556
model_checkpoints/rl_model_44400000_steps.zip  mean_reward:  3.3630009765625  std_reward:  3.670941516491287
model_checkpoints/rl_model_44800000_steps.zip  mean_reward:  3.4493447265625  std_reward:  4.058982794271849
model_checkpoints/rl_model_45200000_steps.zip  mean_reward:  3.4854990234375  std_reward:  3.9969450080233075
model_checkpoints/rl_model_45600000_steps.zip  mean_reward:  3.3291748046875  std_reward:  3.723475480600297
model_checkpoints/rl_model_46000000_steps.zip  mean_reward:  3.3814951171875  std_reward:  4.04699563350482
model_checkpoints/rl_model_46400000_steps.zip  mean_reward:  3.494994140625  std_reward:  4.150435580653127
model_checkpoints/rl_model_46800000_steps.zip  mean_reward:  3.5096611328125  std_reward:  3.806196080663832
model_checkpoints/rl_model_47200000_steps.zip  mean_reward:  3.456333984375  std_reward:  4.009355191617157
model_checkpoints/rl_model_47600000_steps.zip  mean_reward:  3.5380009765625  std_reward:  4.357188904275772
model_checkpoints/rl_model_48000000_steps.zip  mean_reward:  3.6701748046875  std_reward:  4.037960727955904
model_checkpoints/rl_model_48400000_steps.zip  mean_reward:  3.7243330078125  std_reward:  4.265647645426127
model_checkpoints/rl_model_48800000_steps.zip  mean_reward:  3.6461689453125  std_reward:  4.158475610800601
model_checkpoints/rl_model_49200000_steps.zip  mean_reward:  3.6098330078125  std_reward:  4.110776445115815
model_checkpoints/rl_model_49600000_steps.zip  mean_reward:  3.6938310546875  std_reward:  4.15433449678644
model_checkpoints/rl_model_50000000_steps.zip  mean_reward:  3.5841708984375  std_reward:  4.128969607646131
model_checkpoints/rl_model_50400000_steps.zip  mean_reward:  3.559662109375  std_reward:  4.1616356919947926
model_checkpoints/rl_model_50800000_steps.zip  mean_reward:  3.29232421875  std_reward:  4.000451762723516
model_checkpoints/rl_model_51200000_steps.zip  mean_reward:  3.772169921875  std_reward:  4.221172282887116
model_checkpoints/rl_model_51600000_steps.zip  mean_reward:  3.6698427734375  std_reward:  4.259092735564311
model_checkpoints/rl_model_52000000_steps.zip  mean_reward:  3.5719833984375  std_reward:  4.051877454524699
model_checkpoints/rl_model_52400000_steps.zip  mean_reward:  3.6011708984375  std_reward:  4.229864828348732
model_checkpoints/rl_model_52800000_steps.zip  mean_reward:  3.6321708984375  std_reward:  4.18753873702975
model_checkpoints/rl_model_53200000_steps.zip  mean_reward:  3.6960029296875  std_reward:  4.188888697495234
model_checkpoints/rl_model_53600000_steps.zip  mean_reward:  3.743341796875  std_reward:  4.171254101543565
model_checkpoints/rl_model_54000000_steps.zip  mean_reward:  3.68384375  std_reward:  4.1975621776126575
model_checkpoints/rl_model_54400000_steps.zip  mean_reward:  3.5001640625  std_reward:  4.055211961262443
model_checkpoints/rl_model_54800000_steps.zip  mean_reward:  3.6100029296875  std_reward:  4.27771181322906
model_checkpoints/rl_model_55200000_steps.zip  mean_reward:  3.3060048828125  std_reward:  3.9680079629051725
model_checkpoints/rl_model_55600000_steps.zip  mean_reward:  3.607169921875  std_reward:  4.158700850979849
model_checkpoints/rl_model_56000000_steps.zip  mean_reward:  3.625326171875  std_reward:  4.143104131481069
model_checkpoints/rl_model_56400000_steps.zip  mean_reward:  3.65149609375  std_reward:  4.187941629217979
model_checkpoints/rl_model_56800000_steps.zip  mean_reward:  3.4316611328125  std_reward:  4.1525081032908755
model_checkpoints/rl_model_57200000_steps.zip  mean_reward:  3.5908427734375  std_reward:  4.095174798142477
model_checkpoints/rl_model_57600000_steps.zip  mean_reward:  3.5698486328125  std_reward:  4.220756635560582
model_checkpoints/rl_model_58000000_steps.zip  mean_reward:  3.466669921875  std_reward:  4.152667895820806
model_checkpoints/rl_model_58400000_steps.zip  mean_reward:  3.7995087890625  std_reward:  4.121106662958216
model_checkpoints/rl_model_58800000_steps.zip  mean_reward:  3.8329873046875  std_reward:  4.268298897602839
model_checkpoints/rl_model_59200000_steps.zip  mean_reward:  3.8921796875  std_reward:  4.233729161355222
model_checkpoints/rl_model_59600000_steps.zip  mean_reward:  3.7055068359375  std_reward:  4.1599359903865984
model_checkpoints/rl_model_60000000_steps.zip  mean_reward:  3.5329853515625  std_reward:  3.9821947077352076
model_checkpoints/rl_model_60400000_steps.zip  mean_reward:  3.678669921875  std_reward:  4.337849855438532
model_checkpoints/rl_model_60800000_steps.zip  mean_reward:  3.6841689453125  std_reward:  4.230689936985434
model_checkpoints/rl_model_61200000_steps.zip  mean_reward:  3.4616630859375  std_reward:  3.9308279079485673
model_checkpoints/rl_model_61600000_steps.zip  mean_reward:  3.7628427734375  std_reward:  4.099011381353471
model_checkpoints/rl_model_62000000_steps.zip  mean_reward:  3.702484375  std_reward:  4.302920812131268
model_checkpoints/rl_model_62400000_steps.zip  mean_reward:  3.5616630859375  std_reward:  4.056315071342366
model_checkpoints/rl_model_62800000_steps.zip  mean_reward:  3.7746650390625  std_reward:  4.221282281834183
model_checkpoints/rl_model_63200000_steps.zip  mean_reward:  3.6716552734375  std_reward:  4.148113631183609
model_checkpoints/rl_model_63600000_steps.zip  mean_reward:  3.779015625  std_reward:  4.195167646304129
model_checkpoints/rl_model_64000000_steps.zip  mean_reward:  3.57716796875  std_reward:  4.1173206246267195
model_checkpoints/rl_model_64400000_steps.zip  mean_reward:  3.7201689453125  std_reward:  4.242732484860923
model_checkpoints/rl_model_64800000_steps.zip  mean_reward:  3.299845703125  std_reward:  4.258854802746499
model_checkpoints/rl_model_65200000_steps.zip  mean_reward:  3.7330068359375  std_reward:  4.49504755763004
model_checkpoints/rl_model_65600000_steps.zip  mean_reward:  3.79984375  std_reward:  4.257601342285888
model_checkpoints/rl_model_66000000_steps.zip  mean_reward:  3.7609990234375  std_reward:  4.270377084871738
model_checkpoints/rl_model_66400000_steps.zip  mean_reward:  3.615501953125  std_reward:  4.008320455078618
model_checkpoints/rl_model_66800000_steps.zip  mean_reward:  3.4041640625  std_reward:  3.8309012871344246
model_checkpoints/rl_model_67200000_steps.zip  mean_reward:  3.6225048828125  std_reward:  4.07166203478115
model_checkpoints/rl_model_67600000_steps.zip  mean_reward:  3.3725107421875  std_reward:  4.083343777706659
model_checkpoints/rl_model_68000000_steps.zip  mean_reward:  3.6146591796875  std_reward:  4.273909705043986
model_checkpoints/rl_model_68400000_steps.zip  mean_reward:  3.251333984375  std_reward:  3.9173285960405657
model_checkpoints/rl_model_68800000_steps.zip  mean_reward:  3.7968330078125  std_reward:  4.339622092466701
model_checkpoints/rl_model_69200000_steps.zip  mean_reward:  3.7276513671875  std_reward:  4.248604013143629
model_checkpoints/rl_model_69600000_steps.zip  mean_reward:  3.5194970703125  std_reward:  4.275340382568926
model_checkpoints/rl_model_70000000_steps.zip  mean_reward:  3.705345703125  std_reward:  4.203007080797797
model_checkpoints/rl_model_70400000_steps.zip  mean_reward:  3.6143359375  std_reward:  4.101459904036779
model_checkpoints/rl_model_70800000_steps.zip  mean_reward:  3.8239970703125  std_reward:  4.291271563582852
model_checkpoints/rl_model_71200000_steps.zip  mean_reward:  3.5328427734375  std_reward:  4.28868113526003
model_checkpoints/rl_model_71600000_steps.zip  mean_reward:  3.7123232421875  std_reward:  4.190757265492899
model_checkpoints/rl_model_72000000_steps.zip  mean_reward:  3.538333984375  std_reward:  4.0373840284059686
model_checkpoints/rl_model_72400000_steps.zip  mean_reward:  3.9261630859375  std_reward:  4.577076017964797
model_checkpoints/rl_model_72800000_steps.zip  mean_reward:  3.6089951171875  std_reward:  4.143647023491232
model_checkpoints/rl_model_73200000_steps.zip  mean_reward:  3.64551171875  std_reward:  4.204173429398077
model_checkpoints/rl_model_73600000_steps.zip  mean_reward:  3.8159970703125  std_reward:  4.254917838319689
model_checkpoints/rl_model_74000000_steps.zip  mean_reward:  3.649837890625  std_reward:  4.133681093101766
model_checkpoints/rl_model_74400000_steps.zip  mean_reward:  3.7543291015625  std_reward:  4.29702944474644
model_checkpoints/rl_model_74800000_steps.zip  mean_reward:  3.817322265625  std_reward:  4.014499179126382
model_checkpoints/rl_model_75200000_steps.zip  mean_reward:  3.7896787109375  std_reward:  4.2883185924602225
model_checkpoints/rl_model_75600000_steps.zip  mean_reward:  3.589015625  std_reward:  3.9464324097615413
model_checkpoints/rl_model_76000000_steps.zip  mean_reward:  3.7243330078125  std_reward:  4.242701051644116
model_checkpoints/rl_model_76400000_steps.zip  mean_reward:  4.0041728515625  std_reward:  4.273107499120789
model_checkpoints/rl_model_76800000_steps.zip  mean_reward:  3.9421689453125  std_reward:  4.368623307323131
model_checkpoints/rl_model_77200000_steps.zip  mean_reward:  3.6884921875  std_reward:  4.1600567934561035
model_checkpoints/rl_model_77600000_steps.zip  mean_reward:  3.7080087890625  std_reward:  4.106959745826922
model_checkpoints/rl_model_78000000_steps.zip  mean_reward:  3.8560126953125  std_reward:  4.209247041129604
model_checkpoints/rl_model_78400000_steps.zip  mean_reward:  3.736490234375  std_reward:  4.067939754088265
model_checkpoints/rl_model_78800000_steps.zip  mean_reward:  3.7774794921875  std_reward:  4.111039821542196
model_checkpoints/rl_model_79200000_steps.zip  mean_reward:  3.579841796875  std_reward:  4.3159513791711355
model_checkpoints/rl_model_79600000_steps.zip  mean_reward:  3.60633984375  std_reward:  4.134092953788972
model_checkpoints/rl_model_80000000_steps.zip  mean_reward:  3.811162109375  std_reward:  4.188796132732129
model_checkpoints/rl_model_80400000_steps.zip  mean_reward:  3.7906650390625  std_reward:  4.231844313127467
model_checkpoints/rl_model_80800000_steps.zip  mean_reward:  3.6391669921875  std_reward:  4.2943213378892375
model_checkpoints/rl_model_81200000_steps.zip  mean_reward:  3.972998046875  std_reward:  4.22697881863009
model_checkpoints/rl_model_81600000_steps.zip  mean_reward:  3.9868447265625  std_reward:  4.407342094751763
model_checkpoints/rl_model_82000000_steps.zip  mean_reward:  3.5026650390625  std_reward:  4.164644066204525
model_checkpoints/rl_model_82400000_steps.zip  mean_reward:  4.160515625  std_reward:  4.63819827337397
model_checkpoints/rl_model_82800000_steps.zip  mean_reward:  3.6568291015625  std_reward:  4.132398478529773
model_checkpoints/rl_model_83200000_steps.zip  mean_reward:  3.6738447265625  std_reward:  4.330855307240379
model_checkpoints/rl_model_83600000_steps.zip  mean_reward:  3.5888388671875  std_reward:  4.069226931810249
model_checkpoints/rl_model_84000000_steps.zip  mean_reward:  3.8669775390625  std_reward:  4.4296181420737835
model_checkpoints/rl_model_84400000_steps.zip  mean_reward:  3.8943525390625  std_reward:  4.425327300721669
model_checkpoints/rl_model_84800000_steps.zip  mean_reward:  3.7456796875  std_reward:  4.232861064715256
model_checkpoints/rl_model_85200000_steps.zip  mean_reward:  3.9053359375  std_reward:  4.293932518102314
model_checkpoints/rl_model_85600000_steps.zip  mean_reward:  3.8383359375  std_reward:  4.343546842592768
model_checkpoints/rl_model_86000000_steps.zip  mean_reward:  3.85666796875  std_reward:  4.307067751151193
model_checkpoints/rl_model_86400000_steps.zip  mean_reward:  3.7696630859375  std_reward:  4.3681672804583735
model_checkpoints/rl_model_86800000_steps.zip  mean_reward:  3.692830078125  std_reward:  4.1775009515780654
model_checkpoints/rl_model_87200000_steps.zip  mean_reward:  3.7438359375  std_reward:  4.23414891936926
model_checkpoints/rl_model_87600000_steps.zip  mean_reward:  4.0975126953125  std_reward:  4.379901236438636
model_checkpoints/rl_model_88000000_steps.zip  mean_reward:  3.779666015625  std_reward:  4.182327726355715
model_checkpoints/rl_model_88400000_steps.zip  mean_reward:  3.451677734375  std_reward:  3.9504534045490156
model_checkpoints/rl_model_88800000_steps.zip  mean_reward:  4.0165029296875  std_reward:  4.286829982798614
model_checkpoints/rl_model_89200000_steps.zip  mean_reward:  3.740341796875  std_reward:  4.168516344548345
model_checkpoints/rl_model_89600000_steps.zip  mean_reward:  3.7434912109375  std_reward:  4.39755668781159
model_checkpoints/rl_model_90000000_steps.zip  mean_reward:  3.860349609375  std_reward:  4.491243287104138
model_checkpoints/rl_model_90400000_steps.zip  mean_reward:  4.0473388671875  std_reward:  4.349716172810691
model_checkpoints/rl_model_90800000_steps.zip  mean_reward:  3.7415048828125  std_reward:  4.271099164050383
model_checkpoints/rl_model_91200000_steps.zip  mean_reward:  3.7288173828125  std_reward:  4.386709066706771
model_checkpoints/rl_model_91600000_steps.zip  mean_reward:  3.4431708984375  std_reward:  4.05323904352043
model_checkpoints/rl_model_92000000_steps.zip  mean_reward:  3.7225068359375  std_reward:  4.468228090082921
model_checkpoints/rl_model_92400000_steps.zip  mean_reward:  3.6578388671875  std_reward:  4.53910103223099
model_checkpoints/rl_model_92800000_steps.zip  mean_reward:  3.750322265625  std_reward:  4.154841247342079
model_checkpoints/rl_model_93200000_steps.zip  mean_reward:  3.8631787109375  std_reward:  4.303976761920828
model_checkpoints/rl_model_93600000_steps.zip  mean_reward:  4.0856611328125  std_reward:  4.4094245981295135
model_checkpoints/rl_model_94000000_steps.zip  mean_reward:  3.6731748046875  std_reward:  4.3180557774416055
model_checkpoints/rl_model_94400000_steps.zip  mean_reward:  3.7548388671875  std_reward:  4.107619835303536
model_checkpoints/rl_model_94800000_steps.zip  mean_reward:  3.8573447265625  std_reward:  4.272416589363472
model_checkpoints/rl_model_95200000_steps.zip  mean_reward:  3.8678505859375  std_reward:  4.32524045440719
model_checkpoints/rl_model_95600000_steps.zip  mean_reward:  3.5605009765625  std_reward:  4.157579262642652
model_checkpoints/rl_model_96000000_steps.zip  mean_reward:  3.7203349609375  std_reward:  4.204562806442214
model_checkpoints/rl_model_96400000_steps.zip  mean_reward:  3.816328125  std_reward:  4.199969791076648
model_checkpoints/rl_model_96800000_steps.zip  mean_reward:  3.903658203125  std_reward:  4.414790592420664
model_checkpoints/rl_model_97200000_steps.zip  mean_reward:  3.85366796875  std_reward:  4.298118099177042
model_checkpoints/rl_model_97600000_steps.zip  mean_reward:  3.8476630859375  std_reward:  4.213982543747005
model_checkpoints/rl_model_98000000_steps.zip  mean_reward:  3.6726669921875  std_reward:  4.18058532852042
model_checkpoints/rl_model_98400000_steps.zip  mean_reward:  3.6381728515625  std_reward:  4.139645862115554
model_checkpoints/rl_model_98800000_steps.zip  mean_reward:  3.8556630859375  std_reward:  4.259604792664659
model_checkpoints/rl_model_99200000_steps.zip  mean_reward:  3.83933203125  std_reward:  4.294305227151612
model_checkpoints/rl_model_99600000_steps.zip  mean_reward:  3.96217578125  std_reward:  4.302639976562508
model_checkpoints/rl_model_100000000_steps.zip  mean_reward:  3.751826171875  std_reward:  4.220561619252634
model_checkpoints/rl_model_100400000_steps.zip  mean_reward:  3.5991728515625  std_reward:  4.298485667494113
model_checkpoints/rl_model_100800000_steps.zip  mean_reward:  3.7231787109375  std_reward:  4.346162725956933
model_checkpoints/rl_model_101200000_steps.zip  mean_reward:  3.9398388671875  std_reward:  4.1603416245944755
model_checkpoints/rl_model_101600000_steps.zip  mean_reward:  3.8555126953125  std_reward:  4.266438756319112
model_checkpoints/rl_model_102000000_steps.zip  mean_reward:  3.867498046875  std_reward:  4.216866610312296
model_checkpoints/rl_model_102400000_steps.zip  mean_reward:  3.57833203125  std_reward:  4.537609223352307
model_checkpoints/rl_model_102800000_steps.zip  mean_reward:  3.855009765625  std_reward:  4.64098073307198
model_checkpoints/rl_model_103200000_steps.zip  mean_reward:  3.686  std_reward:  4.3843232962948475
model_checkpoints/rl_model_103600000_steps.zip  mean_reward:  3.9156865234375  std_reward:  4.403817691009975
model_checkpoints/rl_model_104000000_steps.zip  mean_reward:  3.8174951171875  std_reward:  4.414176597730176
model_checkpoints/rl_model_104400000_steps.zip  mean_reward:  3.63785546875  std_reward:  4.198577936559183
model_checkpoints/rl_model_104800000_steps.zip  mean_reward:  3.8050029296875  std_reward:  4.250315661956371
model_checkpoints/rl_model_105200000_steps.zip  mean_reward:  3.982173828125  std_reward:  4.462228522442717
model_checkpoints/rl_model_105600000_steps.zip  mean_reward:  3.748501953125  std_reward:  4.246563175124052
model_checkpoints/rl_model_106000000_steps.zip  mean_reward:  3.7605126953125  std_reward:  4.1911727198891935
model_checkpoints/rl_model_106400000_steps.zip  mean_reward:  3.7184990234375  std_reward:  4.408563307266921
model_checkpoints/rl_model_106800000_steps.zip  mean_reward:  3.7833544921875  std_reward:  4.367612164696948
model_checkpoints/rl_model_107200000_steps.zip  mean_reward:  3.6185009765625  std_reward:  4.202457492163499
model_checkpoints/rl_model_107600000_steps.zip  mean_reward:  3.69584765625  std_reward:  4.201775092588331
model_checkpoints/rl_model_108000000_steps.zip  mean_reward:  3.9423369140625  std_reward:  4.459200971643114
model_checkpoints/rl_model_108400000_steps.zip  mean_reward:  3.4854912109375  std_reward:  4.247483852612976
model_checkpoints/rl_model_108800000_steps.zip  mean_reward:  3.897173828125  std_reward:  4.299111042569597
model_checkpoints/rl_model_109200000_steps.zip  mean_reward:  4.1019970703125  std_reward:  4.487798836512418
model_checkpoints/rl_model_109600000_steps.zip  mean_reward:  3.735833984375  std_reward:  4.168302921183663
model_checkpoints/rl_model_110000000_steps.zip  mean_reward:  3.8561630859375  std_reward:  4.234439589508681
model_checkpoints/rl_model_110400000_steps.zip  mean_reward:  3.96599609375  std_reward:  4.205175839035031
model_checkpoints/rl_model_110800000_steps.zip  mean_reward:  4.049998046875  std_reward:  4.531537530528757
model_checkpoints/rl_model_111200000_steps.zip  mean_reward:  3.9081708984375  std_reward:  4.1120213252535756
model_checkpoints/rl_model_111600000_steps.zip  mean_reward:  3.746841796875  std_reward:  4.176336896877082
model_checkpoints/rl_model_112000000_steps.zip  mean_reward:  3.4705087890625  std_reward:  4.39206502014496
model_checkpoints/rl_model_112400000_steps.zip  mean_reward:  4.0691494140625  std_reward:  4.319335713895112
model_checkpoints/rl_model_112800000_steps.zip  mean_reward:  3.7371708984375  std_reward:  4.178844666024181
model_checkpoints/rl_model_113200000_steps.zip  mean_reward:  3.75399609375  std_reward:  4.219035217763854
model_checkpoints/rl_model_113600000_steps.zip  mean_reward:  3.8765029296875  std_reward:  4.287440720360116
model_checkpoints/rl_model_114000000_steps.zip  mean_reward:  3.55683203125  std_reward:  4.1218653902584155
model_checkpoints/rl_model_114400000_steps.zip  mean_reward:  3.9623447265625  std_reward:  4.357542761399607
model_checkpoints/rl_model_114800000_steps.zip  mean_reward:  3.74501171875  std_reward:  4.402099775737355
model_checkpoints/rl_model_115200000_steps.zip  mean_reward:  3.7391728515625  std_reward:  4.072052173221127
model_checkpoints/rl_model_115600000_steps.zip  mean_reward:  3.8206650390625  std_reward:  4.1559494007001305
model_checkpoints/rl_model_116000000_steps.zip  mean_reward:  3.62833984375  std_reward:  4.410613378612445
model_checkpoints/rl_model_116400000_steps.zip  mean_reward:  3.9911845703125  std_reward:  4.413496604984573
model_checkpoints/rl_model_116800000_steps.zip  mean_reward:  4.0731630859375  std_reward:  4.450779288607877
model_checkpoints/rl_model_117200000_steps.zip  mean_reward:  3.9106640625  std_reward:  4.229736702076552
model_checkpoints/rl_model_117600000_steps.zip  mean_reward:  3.852166015625  std_reward:  4.30571653320856
model_checkpoints/rl_model_118000000_steps.zip  mean_reward:  4.0095078125  std_reward:  4.565562985844798
model_checkpoints/rl_model_118400000_steps.zip  mean_reward:  3.9303330078125  std_reward:  4.247733128239526
model_checkpoints/rl_model_118800000_steps.zip  mean_reward:  3.9781630859375  std_reward:  4.500185383970723
model_checkpoints/rl_model_119200000_steps.zip  mean_reward:  3.842169921875  std_reward:  4.265662835824699
model_checkpoints/rl_model_119600000_steps.zip  mean_reward:  3.595337890625  std_reward:  4.198250973841465
model_checkpoints/rl_model_120000000_steps.zip  mean_reward:  4.11433203125  std_reward:  4.556207569298783
model_checkpoints/rl_model_120400000_steps.zip  mean_reward:  3.874494140625  std_reward:  4.292750807049106
model_checkpoints/rl_model_120800000_steps.zip  mean_reward:  3.9191572265625  std_reward:  4.351924048031743
model_checkpoints/rl_model_121200000_steps.zip  mean_reward:  3.90633203125  std_reward:  4.317143323060694
model_checkpoints/rl_model_121600000_steps.zip  mean_reward:  3.6546611328125  std_reward:  4.211495616027763
model_checkpoints/rl_model_122000000_steps.zip  mean_reward:  3.5515234375  std_reward:  4.257922394385279
model_checkpoints/rl_model_122400000_steps.zip  mean_reward:  3.8263359375  std_reward:  4.131694157600939
model_checkpoints/rl_model_122800000_steps.zip  mean_reward:  3.685173828125  std_reward:  4.252494823858125
model_checkpoints/rl_model_123200000_steps.zip  mean_reward:  3.802154296875  std_reward:  4.318442379714039
model_checkpoints/rl_model_123600000_steps.zip  mean_reward:  3.772501953125  std_reward:  4.247317154934259
model_checkpoints/rl_model_124000000_steps.zip  mean_reward:  3.931826171875  std_reward:  4.298827593447814
model_checkpoints/rl_model_124400000_steps.zip  mean_reward:  3.7238408203125  std_reward:  4.2101842499270745
model_checkpoints/rl_model_124800000_steps.zip  mean_reward:  4.1441767578125  std_reward:  4.6573355730829
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

Self-play with dynamic sampling and evaluation  (4 sequence of transaction history) 20% EV Agent, reward not modified
	MLPPolicy ReLU
model_checkpoints/rl_model_400000_steps  mean_reward:  -0.005880859375  std_reward:  0.1475048100479241
model_checkpoints/rl_model_800000_steps  mean_reward:  -0.0703583984375  std_reward:  1.0630081706370023
model_checkpoints/rl_model_1200000_steps  mean_reward:  -0.07485546875  std_reward:  0.487150648969351
model_checkpoints/rl_model_1600000_steps  mean_reward:  -0.12873828125  std_reward:  2.7586434465192857
model_checkpoints/rl_model_2000000_steps  mean_reward:  0.022146484375  std_reward:  1.910323566949739
model_checkpoints/rl_model_2400000_steps  mean_reward:  0.112193359375  std_reward:  1.601991207449902
model_checkpoints/rl_model_2800000_steps  mean_reward:  0.1079189453125  std_reward:  1.6857844407284324
model_checkpoints/rl_model_3200000_steps  mean_reward:  0.17445703125  std_reward:  1.6504367046035755
model_checkpoints/rl_model_3600000_steps  mean_reward:  0.14755078125  std_reward:  1.6178256153538138
model_checkpoints/rl_model_4000000_steps  mean_reward:  0.2559443359375  std_reward:  1.8772881666551395
model_checkpoints/rl_model_4400000_steps  mean_reward:  0.181890625  std_reward:  1.93051298638723
model_checkpoints/rl_model_4800000_steps  mean_reward:  0.1311025390625  std_reward:  1.7829999180939227
model_checkpoints/rl_model_5200000_steps  mean_reward:  0.43001953125  std_reward:  2.3999380207800813
model_checkpoints/rl_model_5600000_steps  mean_reward:  0.2679150390625  std_reward:  2.5705048637364527
model_checkpoints/rl_model_6000000_steps  mean_reward:  0.2784091796875  std_reward:  2.6715068703128817
model_checkpoints/rl_model_6400000_steps  mean_reward:  0.26623046875  std_reward:  3.633428090904299
model_checkpoints/rl_model_6800000_steps  mean_reward:  0.2052998046875  std_reward:  2.7052595045574757
model_checkpoints/rl_model_7200000_steps  mean_reward:  0.5957802734375  std_reward:  3.749588268661699
model_checkpoints/rl_model_7600000_steps  mean_reward:  0.752859375  std_reward:  3.984961351618383
model_checkpoints/rl_model_8000000_steps  mean_reward:  0.536693359375  std_reward:  5.275438738699265
model_checkpoints/rl_model_8400000_steps  mean_reward:  0.7269189453125  std_reward:  3.9009389113643915
model_checkpoints/rl_model_8800000_steps  mean_reward:  0.6406396484375  std_reward:  4.63870239245461
model_checkpoints/rl_model_9200000_steps  mean_reward:  0.64665625  std_reward:  4.570854199566338
model_checkpoints/rl_model_9600000_steps  mean_reward:  0.4194091796875  std_reward:  3.3663532623667685
model_final  							  mean_reward:  0.6919052734375  std_reward:  4.766991596116788
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
