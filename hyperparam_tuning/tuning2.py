"""
This script trains the agent and saves resulting model as SAVE_NAME.zip. It also saves copies of past models to model_checkpoints folder.
"""
import tensorflow as tf
import optuna
import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback,  BaseCallback, EveryNTimesteps
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
import time


	
def make_env(rank, agents, seed=10000):
	"""
	Utility function for multiprocessed env.
	"""
	def _init():
		env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE)
		env.seed(seed + rank)
		return env
	set_global_seeds(seed)
	return _init

class CustomCallback(BaseCallback):
	"""
	A custom callback that derives from ``BaseCallback``.
	"""
	def __init__(self, model, env, verbose=0):
		super(CustomCallback, self).__init__(verbose)
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

def optimize_ppo2(trial):
	""" Learning hyperparamters we want to optimise"""
	return {
		'n_steps': int(trial.suggest_loguniform('n_steps', 64, 2048)),
		'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
		'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
		'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
		'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
		'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
		'lam': trial.suggest_uniform('lam', 0.8, 1.)
	}

	
def optimize_agent(trial):
	""" Train the model and optimize
		Optuna maximises the negative log likelihood, so we
		need to negate the reward here
	"""
	model_params = optimize_ppo2(trial)
	
	"""
	env = SubprocVecEnv([make_env(i, agents) for i in range(num_cpu)])
	model = PPO2(POLICY_TYPE, env, nminibatches=1, **model_params) 
	# n_steps (int) – The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
	# by default n_steps=128. After 128 steps for each env, the policy will be updated. If 3 days per game and 2 seq per day, then every update reqires 128/2/3 = 21 games
	env.env_method("set_model_reference", model.get_parameters())
	"""
	env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE)
	model = PPO2(POLICY_TYPE, env, nminibatches=1, **model_params) 
	env.set_model_reference(model.get_parameters())
	
	# save a copy of model every 5e4*num_cpu games
	copy_call_back = CustomCallback(model, env)
	call_back_list = [EveryNTimesteps(n_steps=model_params['n_steps']*10, callback=copy_call_back)]

	model.learn(total_timesteps=TRAINING_TIME_STEPS, callback=call_back_list)
	
	# Evaluate the result against baseline agent
	env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
		seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
		random_seq = True, self_play = False, obs_transaction_history_size=TRANSACTION_HISTORY_SIZE,
		eval=True)

	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES)
	
	with open("optuna_params2/"+str(trial.number)+".txt", "w") as file:
		# Writing data to a file
		file.write("mean reward: " + str(mean_reward) + "	std reward: " + str(std_reward) +"\n")
		file.write(str(model_params))
	
	return -1 * mean_reward
	
if __name__ == '__main__':
	# starts with a simpler version of the game
	NUM_PLAYERS = 2
	SEQ_PER_DAY = 2
	CARDS_PER_SUIT = 10
	SUIT_COUNT = 1
	BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
	POLICY_TYPE = 'MlpPolicyReLU'
	SELF_COPY_FREQ = 10 # copy the agent itself to past selves bank every 10 policy updates
	SELF_PLAY = True
	EVAL_FREQ = int(1e5)
	EVAL_EPISODES = int(100)
	SAVE_NAME = "model_final"
	# num_cpu = 1  # Number of processes to use. It is set to 8 to get more out of 8 threads
	TRAINING_TIME_STEPS = (int)(8e5)
	TRANSACTION_HISTORY_SIZE =4 # one sequence of transaction. Which is 1*4*3 elements in a 3-player game

	HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

	PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

	# add 1 baseline agent
	agents = []
	agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))
	#agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))


	study = optuna.create_study()
	study.optimize(optimize_agent, n_trials=100, n_jobs=8)