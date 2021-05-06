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
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env

def make_env(rank, agents, seed=10000):
	"""
	Utility function for multiprocessed env.
	"""
	def _init():
		env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
			seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
			random_seq = True, self_play = SELF_PLAY, policy_type = POLICY_TYPE, self_copy_freq = SELF_COPY_FREQ,
			obs_transaction_history_size=TRANSACTION_HISTORY_SIZE, n_steps=N_STEPS)
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


if __name__ == '__main__':
	# starts with a simpler version of the game
	NUM_PLAYERS = 2
	SEQ_PER_DAY = 2
	CARDS_PER_SUIT = 10
	SUIT_COUNT = 1
	BETTING_MARGIN = CARDS_PER_SUIT*CARDS_PER_SUIT/100
	POLICY_TYPE = 'MlpPolicyReLU128'
	SELF_COPY_FREQ = 10 # copy the agent itself to past selves bank every 10 policy updates
	SELF_PLAY = True
	EVAL_FREQ = int(1e5)
	EVAL_EPISODES = int(1e2)
	SAVE_NAME = "model_final"
	num_cpu = 8  # Number of processes to use. It is set to 8 to get more out of 8 threads
	TRAINING_TIME_STEPS = (int)(1e7)
	TRANSACTION_HISTORY_SIZE =4 # one sequence of transaction. Which is 1*4*3 elements in a 3-player game

	HAND_COUNT = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)

	PUBLIC_CARDS_COUNT = CARDS_PER_SUIT*SUIT_COUNT - HAND_COUNT*NUM_PLAYERS

	model_params = {'n_steps': int(1403/num_cpu), 'gamma': 0.9378697782327615, 'learning_rate': 0.0002743310803336785, 'ent_coef': 2.2312682753757416e-05, 'cliprange': 0.12718794371596698, 'noptepochs': 32, 'lam': 0.894837193141085}
	N_STEPS = model_params['n_steps']
	
	# add 1 baseline agent
	agents = []
	agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))
	#agents.append(baseline_agents.EVAgent(agent_idx = 1, num_players = NUM_PLAYERS, betting_margin = BETTING_MARGIN, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT, public_cards_count = PUBLIC_CARDS_COUNT))


	# Create the vectorized environment to run multiple game environments in parallel
	env = SubprocVecEnv([make_env(i, agents) for i in range(num_cpu)])


	model = PPO2(POLICY_TYPE, env, **model_params) 
	# n_steps (int) – The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
	# by default n_steps=128. After 128 steps for each env, the policy will be updated. If 3 days per game and 2 seq per day, then every update reqires 128/2/3 = 21 games
	env.env_method("set_model_reference", model.get_parameters())

	# If the environment don't follow the interface, an error will be thrown
	#check_env(env, warn=True)
	#print("The environment is valid.")
	#env.reset_model_bank()

	# save a copy of model every 5e4*num_cpu games
	copy_call_back = CustomCallback(model)
	call_back_list = [CheckpointCallback(save_freq=int(5e4), save_path='./model_checkpoints/'), 
		EveryNTimesteps(n_steps=N_STEPS*10*num_cpu, callback=copy_call_back)]



	model.learn(total_timesteps=TRAINING_TIME_STEPS, callback=call_back_list)


	# save final model
	model.save(SAVE_NAME)


	# Evaluate the result against baseline agent
	env = TradingGameEnv.TradingGameEnv(player_count = NUM_PLAYERS, other_agent_list = agents,
		seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, player_hand_count = HAND_COUNT,
		random_seq = True, self_play = False, obs_transaction_history_size=TRANSACTION_HISTORY_SIZE,
		eval=True)

	print("\n Evaluate the result against baseline agent")
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES)
	print("mean_reward: ", mean_reward)
	print("std_reward: ", std_reward)

	# Playing test rounds
	print("\nPlaying test rounds:")
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
