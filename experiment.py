"""
This script trains the agent and saves resulting model as SAVE_NAME.zip. It also saves copies of past models to model_checkpoints folder.
"""

import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback,  BaseCallback, EveryNTimesteps
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env

def make_env(rank, agents, seed=10000):
	"""
	Utility function for multiprocessed env.
	"""
	def _init():
		env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, n_steps=N_STEPS, 
			EVAgent_percentage= EVAgent_percentage, dynamic_eval = DYNAMIC_EVAL)
		env.seed(seed + rank)
		return env
	set_global_seeds(seed)
	return _init

class CustomCallback(BaseCallback):
	"""
	A custom callback that derives from ``BaseCallback``.
	"""
	def __init__(self, model, verbose=0):
		super(CustomCallback, self).__init__(verbose)
		self.model = model

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		self.model.get_env().env_method("set_model_reference", self.model.get_parameters())
		print("current timestep", self.num_timesteps)
		return True
		
class CustomCallbackSingle(BaseCallback):
	"""
	A custom callback that derives from ``BaseCallback``.
	"""
	def __init__(self, model, env, verbose=0):
		super(CustomCallbackSingle, self).__init__(verbose)
		self.model = model
		self.env = env

	def _on_step(self) -> bool:
		"""
		This method will be called by the model after each call to `env.step()`.

		For child callback (of an `EventCallback`), this will be called
		when the event is triggered.

		:return: (bool) If the callback returns False, training is aborted early.
		"""
		#self.model.get_env().env_method("set_model_reference", self.model.get_parameters())
		self.env.set_model_reference(self.model.get_parameters())
		print("current timestep", self.num_timesteps)
		return True
		
def run_exp():
	# multi process env
	if num_cpu > 1:
		# Create the vectorized environment to run multiple game environments in parallel
		env = SubprocVecEnv([make_env(i, agents) for i in range(num_cpu)])


		model = PPO2(POLICY_TYPE, env, **model_params) 
		# n_steps (int) – The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
		# by default n_steps=128. After 128 steps for each env, the policy will be updated. If 3 days per game and 2 seq per day, then every update reqires 128/2/3 = 21 games
		env.env_method("set_model_reference", model.get_parameters())


		# save a copy of model every 5e4*num_cpu games
		copy_call_back = CustomCallback(model)
		call_back_list = [CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_PATH), 
			EveryNTimesteps(n_steps=N_STEPS*SELF_COPY_FREQ*num_cpu, callback=copy_call_back)]
		
	else:
		# single process env
		env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, n_steps=N_STEPS, 
			EVAgent_percentage= EVAgent_percentage, dynamic_eval = DYNAMIC_EVAL)
		model = PPO2(POLICY_TYPE, env, nminibatches=1, **model_params) 
		env.set_model_reference(model.get_parameters())
		
		# save a copy of model every 5e4*num_cpu games
		copy_call_back = CustomCallbackSingle(model, env)
		call_back_list = [CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_PATH), 
			EveryNTimesteps(n_steps=N_STEPS*SELF_COPY_FREQ*num_cpu, callback=copy_call_back)]
	
	# training
	model.learn(total_timesteps=TRAINING_TIME_STEPS, callback=call_back_list)
	# saving
	model.save(SAVE_PATH+ "rl_model_"+str(TRAINING_TIME_STEPS)+"_steps")
	
def run_exp_dummy_vec():
	# multi process env

	# Create the vectorized environment to run multiple game environments in parallel
	env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, n_steps=N_STEPS, 
			EVAgent_percentage= EVAgent_percentage, dynamic_eval = DYNAMIC_EVAL, dummy_vec = True)
	print(env)
	print("obs space", env.observation_space)
	print("action space", env.action_space.sample())
	env = DummyVecEnv([lambda: env])
	print(env)
	print("obs space", env.observation_space)
	print("action space", env.action_space.sample())


	model = PPO2(POLICY_TYPE, env, nminibatches=1, **model_params) 
	# n_steps (int) – The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
	# by default n_steps=128. After 128 steps for each env, the policy will be updated. If 3 days per game and 2 seq per day, then every update reqires 128/2/3 = 21 games
	env.env_method("set_model_reference", model.get_parameters())


	# save a copy of model every 5e4*num_cpu games
	copy_call_back = CustomCallback(model)
	call_back_list = [CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_PATH), 
		EveryNTimesteps(n_steps=N_STEPS*SELF_COPY_FREQ*num_cpu, callback=copy_call_back)]
		

	# training
	model.learn(total_timesteps=TRAINING_TIME_STEPS, callback=call_back_list)
	# saving
	model.save(SAVE_PATH+ "rl_model_"+str(TRAINING_TIME_STEPS)+"_steps")

def print_params():
	print("**********", POLICY_TYPE, EVAgent_percentage, SELF_PLAY, TRANSACTION_HISTORY_SIZE, SELF_COPY_FREQ,
		DYNAMIC_EVAL, NUM_PLAYERS, CARDS_PER_SUIT, "**********")


if __name__ == '__main__':
	# default params
	NUM_PLAYERS = 2
	SEQ_PER_DAY = 2
	CARDS_PER_SUIT = 10
	SUIT_COUNT = 1
	BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
	POLICY_TYPE = 'MlpPolicyRelu64_3'
	SELF_COPY_FREQ = 10 # copy the agent itself to past selves bank every 10 policy updates
	SELF_PLAY = True
	DYNAMIC_EVAL = True
	EVAgent_percentage = 0.2
	num_cpu = 1  # Number of processes to use. It is set to 8 to get more out of 8 threads
	TRAINING_TIME_STEPS = (int)(1e6)
	TRANSACTION_HISTORY_SIZE =4 # one sequence of transaction. Which is 1*4*3 elements in a 3-player game
	SAVE_FREQ = int(1e5/num_cpu)
	
	HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

	PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

	model_params = {'n_steps': int(1400/num_cpu), 'gamma': 0.9378697782327615, 'learning_rate': 0.0002743310803336785, 'ent_coef': 2.2312682753757416e-05, 'cliprange': 0.12718794371596698, 'noptepochs': 32, 'lam': 0.894837193141085}
	N_STEPS = model_params['n_steps']
	
	# add 1 baseline agent
	agents = []
	agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))


	# varying params
	
	"""
	# plot net arch
	num_layers = [2, 3, 4]
	num_neurons = [64, 128]
	acts = ["Relu", "Tanh"]
	for num_layer in num_layers:
		for num_neuron in num_neurons:
			for act in acts:
				POLICY_TYPE = 'MlpPolicy'+str(act)+str(num_neuron)+"_"+str(num_layer) # e.g. MlpPolicyRelu64_2
				SAVE_PATH= './model_checkpoints/plot_net_arch/' + POLICY_TYPE +'/'
				print_params()
				run_exp()
	"""
	
	"""
	# % of EVAgent
	percent = [0, 10, 20, 30, 40, 60, 80, 100]
	for p in percent:
		EVAgent_percentage = p/100
		SAVE_PATH= './model_checkpoints/plot_EVAgent_percent/' + str(p) +'/'
		
		if p == 100:
			# no self play
			SELF_PLAY = False
			
		print_params()
		run_exp()
	"""
	
	"""
	# plot different transaction history length
	length = [0, 1, 2, 4, 6, 8]
	for l in length:
		TRANSACTION_HISTORY_SIZE = l
		SAVE_PATH= './model_checkpoints/plot_trans_hist_size/' + str(l) +'/'
			
		print_params()
		run_exp()
	"""
	
	"""
	# plot model bank update frequency
	freq = [1, 5, 10, 100, 1000]
	for f in freq:
		SELF_COPY_FREQ = f
		SAVE_PATH= './model_checkpoints/plot_model_bank_update_freq/' + str(f) +'/'
			
		print_params()
		run_exp()
	"""
	
	"""
	# dynamic sampling and evaluation (on and off)
	dynamic = [False]
	for d in dynamic:
		DYNAMIC_EVAL = d
		SAVE_PATH= './model_checkpoints/plot_dynamic_eval/' + str(d) +'/'
			
		print_params()
		run_exp()
	"""
	
	"""
	# more cards
	# cards = [20, 40, 60,  80 , 100]	# 150, 200
	cards = [40]
	for c in cards:
		POLICY_TYPE = 'MlpPolicyReLU10000'
		SAVE_PATH= './model_checkpoints/plot_cards/MlpPolicyReLU10000' + str(c) +'/'
		#POLICY_TYPE = 'MlpPolicyRelu512_3'
		#SAVE_PATH= './model_checkpoints/plot_cards/' + str(c) +'/'
		
		CARDS_PER_SUIT = c
		BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
		HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

		PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

		
		# add 1 baseline agent
		agents = []
		agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

		print_params()
		run_exp()
	"""

	
	# more cards
	# cards = [20, 40, 60,  80 , 100]	# 150, 200
	cards = [10, 20, 40, 60]	# 80 , 100, 150, 200
	for c in cards:
		POLICY_TYPE = 'MlpLstmPolicy'
		SAVE_PATH= './model_checkpoints/plot_cards/MlpLstmPolicy' + str(c) +'/'
		#POLICY_TYPE = 'MlpPolicyRelu512_3'
		#SAVE_PATH= './model_checkpoints/plot_cards/' + str(c) +'/'
		
		CARDS_PER_SUIT = c
		BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
		HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

		PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

		
		# add 1 baseline agent
		agents = []
		agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

		print_params()
		# run_exp()
		run_exp_dummy_vec()
	
	"""
	for c in cards:
		POLICY_TYPE = 'LstmPolicy'
		SAVE_PATH= './model_checkpoints/plot_cards/LstmPolicy' + str(c) +'/'
		#POLICY_TYPE = 'MlpPolicyRelu512_3'
		#SAVE_PATH= './model_checkpoints/plot_cards/' + str(c) +'/'
		
		CARDS_PER_SUIT = c
		BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
		HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

		PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

		
		# add 1 baseline agent
		agents = []
		agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

		print_params()
		# run_exp()
		run_exp_dummy_vec()
	"""

	"""
	# more players
	players = [2, 3, 4, 5, 6, 10] #6, 10
	for p in players:
		POLICY_TYPE = 'MlpPolicyRelu512_3'
		SAVE_PATH= './model_checkpoints/plot_players/' + str(p) +'/'
		NUM_PLAYERS = p
		CARDS_PER_SUIT = 30
		BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
		HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

		PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

		
		# add baseline agents
		agents = []
		for i in range(p-1):
			agents.append(baseline_agents.EVAgent(agent_idx = i+1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))

		print_params()
		run_exp()
	"""