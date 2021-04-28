import gym
from gym import spaces
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import math
from queue import PriorityQueue
from stable_baselines import PPO2


INITIAL_ACCOUNT_BALANCE = 10000
AGENT_INDEX = 0
NO_ACTION_PENALTY = -5


def softmax(arr, axis=None):
	z = arr - arr.max(axis=axis, keepdims=True)
	e_z = np.exp(z)
	return e_z / e_z.sum(axis=axis, keepdims=True)
	
	
class TradingGameEnv(gym.Env):
	"""A trading game environment for OpenAI gym"""
	metadata = {'render.modes': ['human']}

	"""
	parameters:
		self_play: use self-play training mode. A copy of self policy will be used as the opponent.
	"""
	def __init__(self, player_count = 2, suit_count = 1, number_of_sub_piles = 4, other_agent_list = [],
		seq_per_day = 1, random_seq = False, cards_per_suit = 13, public_cards_count = 9, extensive_obs = False,
		self_play = False, policy_type = 'MlpPolicy', self_copy_freq = -1, model_quality_lr=0.01):

		super(TradingGameEnv, self).__init__()
		self.player_count = player_count
		self.suit_count = suit_count
		self.number_of_sub_piles = number_of_sub_piles
		self.other_agents = other_agent_list
		self.random_seq = random_seq
		self.cards_per_suit = cards_per_suit
		self.seq_per_day = seq_per_day
		self.SUIT_SUM = (1+cards_per_suit)*cards_per_suit/2
		self.turn_sequence = np.arange(0, self.player_count)
		self.public_cards_count = public_cards_count
		self.self_play = self_play
		self.game_count = 0
		self.games_per_update = int(128/self.seq_per_day/(self.number_of_sub_piles-1)) + 1
		self.observation_memory = 1
		self.transaction_history = np.zeros((self.observation_memory, self.player_count, 4))
		self.transaction_history_size = self.transaction_history.shape[0]*self.transaction_history.shape[1]*self.transaction_history.shape[2]

		# variables for self play training
		if self_play:
			self.model_bank = []	# a bank that holds all past selves
			self.model_qualities = np.array([], dtype=np.float32)	# represent the quality of the models in the bank.
			self.model_probabilities = None
			self.policy_type = policy_type
			self.current_opponents_index = None # index of current opponents in the bank
			self.self_copy_freq = self_copy_freq # add the current agent to the bank every self_copy_freq updates
			self.model = None
			self.model_quality_lr = model_quality_lr

		if len(self.other_agents) != self.player_count-1:
			raise Exception("Error: other_agent_list do not conform to the number of players. You may need to add/remove some other agents")

		if self_play and policy_type != 'MlpPolicy':
			print("self_play policy type ", policy_type, " may not be implemented")


		# Actions: buy (1=true), sell, price for buy, price for sell
		# action: [1, 0, 23, 25]
		self.action_space =  spaces.MultiDiscrete([2, 2, self.SUIT_SUM, self.SUIT_SUM])

		# about half cards are reserved for the public pile
		# each player is given player_hand_count private cards
		self.player_hand_count = (int) ((self.cards_per_suit)/2*self.suit_count/player_count)


		# sub-piles
		self.public_sub_pile_cards_count = math.ceil(self.public_cards_count/self.number_of_sub_piles)

		# observation: revealed public pile + own hand  + balance + contract

		# first public_cards_count cards are public pile
		# then self.player_hand_count is own_hand
		# [public pile + own hand]
		# 6 cards for public pile, 3 cards in own hand
		# [1, 5, 0, 0, 0, 0, 2, 11, 12]
		#Last numbers are own hand.
		self.obs_element_count = self.public_cards_count + self.player_hand_count + self.transaction_history_size
		lower = np.zeros(self.obs_element_count, dtype=np.int) #[0 0 0 0 0]
		#upper = np.full(self.obs_element_count, self.cards_per_suit, dtype=np.int)
		upper = np.full(self.obs_element_count, self.SUIT_SUM, dtype=np.int)
		'''
		what if they buy more than suit_sum contracts?
		'''
		self.observation_space = spaces.Box(lower , upper, dtype=np.int)
		### TODO: Modify size of obs space.

		if self.self_play:
			self.init_self_play_opponents()

	# add a copy of past self to the bank
	# initialize the quality of the past self to max quality, i.e. 1
	def add_past_self(self, model_param):
		self.model_bank.append(model_param)
		self.model_qualities = np.append(self.model_qualities, 1)

	# print params of the policy network
	def print_self_param(self, key = None):
		if key != None:
			print(self.model[key])
		else:
			print(self.model)

	# add a reference of the policy model
	def set_model_reference(self, model):
		self.model = model

	# init other opponent list
	def init_self_play_opponents(self):
		self.other_agents = []
		# there are player_count - 1 opponents
		for i in range(self.player_count - 1):
			self.other_agents.append(PPO2(self.policy_type, self))

	# reset self play opponents every game
	def reset_self_play_opponents(self):
		# if it's not first game, update opponent model quality after a game (dynamic sampling and evaluation)
		if self.game_count != 0:
			# check if the agent defeats opponents based upon net worth
			net_worth = self.balance + self.contract * self.public_pile.sum()
			# print("net_worth", net_worth)
			for i in range(len(self.other_agents)):
				# if the net worth of opponent is less than the agent, reduce the quality score of the opponent
				if net_worth[AGENT_INDEX] > net_worth[i+1]:
					# print("win")
					model_idx = self.current_opponents_index[i]
					# the amount to reduce = model_quality_lr / (total number of models * probability of choosing this model)
					self.model_qualities[model_idx] -= self.model_quality_lr/(len(self.model_qualities) * self.model_probabilities[model_idx])
				

		# add the model itself to the bank every self_copy_freq updates
		if self.game_count == 0 or self.game_count % (self.self_copy_freq*self.games_per_update) == 0:
			self.add_past_self(self.model)
			#self.print_self_param(key = 'model/pi/b:0')

		# choose new opponents
		self.model_probabilities = softmax(self.model_qualities)
		self.current_opponents_index = []
		
		for i in range(self.player_count-1):
			if random.random() < 0.8:
				# 80% of time play against immediate past self
				self.current_opponents_index = [-1]
				self.other_agents[i].load_parameters(self.model_bank[-1])
				self.model_probabilities = softmax(self.model_qualities)
			else:
				# 20% of time play against random past self
				
				# choose past self based upon quality of the past selves using softmax probability distribution
				# i.e. a self that has better quality is more likely to get selected
				chosen_index = np.random.choice(np.arange(len(self.model_bank)), 1, replace=False, p=self.model_probabilities)[0]
				
				# save the chosen index and load model
				self.current_opponents_index.append(chosen_index)
				self.other_agents[i].load_parameters(self.model_bank[chosen_index])
				
		# print("game ", self.game_count)
		# print("model bank", np.arange(len(self.model_bank)))
		# print("qualities", self.model_qualities)
		# print("softmax", self.model_probabilities)
		# print("idx", self.current_opponents_index, "\n\n")

	def reset_model_bank(self):
		self.game_count = 0
		self.model_bank = []	# a bank that holds all past selves
		self.model_qualities = np.array([], dtype=np.float32)	# represent the quality of the models in the bank.
		self.model_probabilities = None
		self.current_opponents_index = None # index of current opponents in the bank

	def reset(self):
		# update quality of current opponents and choose new opponents if in self play mode
		if self.self_play:
			self.reset_self_play_opponents()

		# Reset the state of the environment to an initial state
		# the agent is place at AGENT_INDEX = 0 row
		self.balance = np.full(self.player_count, INITIAL_ACCOUNT_BALANCE, dtype=np.float32) #[agent, other_agent1, other_agent2, ....]
		self.contract = np.zeros(self.player_count, dtype=np.int)
		self.transaction_history = np.zeros((self.observation_memory, self.player_count, 4))
		self.sequence_counter = 1
		self.day = 1
		self.total_reward = 0
		self.game_count += 1

		if self.random_seq:
			np.random.shuffle(self.turn_sequence)

		suit_list = np.arange(1, self.cards_per_suit+1)
		np.random.shuffle(suit_list)

		self.hands = suit_list[0:self.player_count * self.player_hand_count].reshape((self.player_count, self.player_hand_count))


		self.public_pile = suit_list[self.player_count * self.player_hand_count:]


		self.sell_offer = PriorityQueue()
		self.buy_offer = PriorityQueue()


		return self._next_observation()

	def step(self, action):
		### TODO: Use data structures to record transaction history
		##

		self.contract_prev = self.contract[AGENT_INDEX]
		self.balance_prev = self.balance[AGENT_INDEX]

		# take actions based upon turn sequence
		for i in self.turn_sequence:

			# if it is an opponent agent
			if i != AGENT_INDEX:
				obs_i = self._next_observation(i)
				action_i = self.other_agents[i-1].predict(obs_i)

				if self.self_play:
					action_i = action_i[0]
				self.transaction_history[0][i] = action_i
				'''
				### TODO:
					(1)Infinite Transaction history, pass last X elements to observation space.
					(2)Create a Queue for transaction history
				'''
				'''
				actions of agent i takes place here
				'''
				self._take_action(action_i, i)

			else:
				# if it is our agent

				# Execute one time step within the environment
				self.transaction_history[0][i] = action
				self._take_action(action, AGENT_INDEX)

		'''
		# compute reward
		# reward = expected profit * day
		#  = (expected value of public pile * amount of contract + balance  – initial balance ) * day
		#  expected value of public pile =
		#     (revealed public pile + expected un-revealed public card value * un-revealed card count)
		revealed_card_index = min(self.day*self.public_sub_pile_cards_count, self.public_pile.shape[0])
		revealed_value = self.public_pile[0:revealed_card_index].sum()

		# expected un-revealed public card value = (sum of all cards
		# 			– revealed public piles – own hand)/(remaining card count)
		expected_card_value = ((self.SUIT_SUM - revealed_value - self.hands[AGENT_INDEX,:].sum()) /
							(self.cards_per_suit - self.player_hand_count - revealed_card_index))
		expected_value_of_public_pile = (revealed_value + expected_card_value  *
							(self.public_pile.shape[0] - revealed_card_index))

		reward = (expected_value_of_public_pile * self.contract[AGENT_INDEX] +
							self.balance[AGENT_INDEX] - INITIAL_ACCOUNT_BALANCE)
		'''

		# give reward based on ground truth for last action only
		reward = (self.public_pile.sum() * (self.contract[AGENT_INDEX]-self.contract_prev) +
							(self.balance[AGENT_INDEX]-self.balance_prev))

		# add some panelty if the agent is doing nothing
		if (action[0] == 0 and action[1] == 0):
			reward += NO_ACTION_PENALTY
		# reward *= self.day

		"""
		# if it's the last day, give final reward
		if self.day == self.number_of_sub_piles-1 and self.sequence_counter == self.seq_per_day:
			final_reward = ((self.public_pile.sum() * self.contract[AGENT_INDEX] +
							self.balance[AGENT_INDEX] - INITIAL_ACCOUNT_BALANCE)*
							(self.day + 1))
			reward += final_reward

		self.total_reward += reward
		"""

		# if it is the last sequence of that day, go to next day
		if self.sequence_counter == self.seq_per_day:
			self.day += 1

			#reset sequence number
			self.sequence_counter = 1

			# clear the offers for the day
			with self.sell_offer.mutex:
				self.sell_offer.queue.clear()
			with self.buy_offer.mutex:
				self.buy_offer.queue.clear()

			# randomize sequence for next day
			if self.random_seq:
				np.random.shuffle(self.turn_sequence)
		else:
			self.sequence_counter += 1

		done = self.day > self.number_of_sub_piles-1
		obs = self._next_observation()



		return obs, (float) (reward), done, {}

	def _next_observation(self, agent_index = AGENT_INDEX):
		# get the observation for the agent at the index
		## TODO: Populate observation array.
		obs = np.zeros(self.obs_element_count, dtype=np.int)

		# public pile
		revealed_card_index = min(self.day*self.public_sub_pile_cards_count, self.public_pile.shape[0])
		obs[0:revealed_card_index] = self.public_pile[0:revealed_card_index]

		# own hand
		obs[self.public_cards_count:self.public_cards_count + self.transaction_history_size] = self.transaction_history.reshape(-1)


		obs[self.public_cards_count + self.transaction_history_size:] = self.hands[agent_index, :]

		# return [1, 5, 0, 0, 0, 0,***transaction_history*** ,2, 11, 12]
		return obs

	def get_next_offer(self, agent, buy):
		# get next ofer that is not placed by the agent
		# return: offer: next highest buy/ lowest sell offer that is not from the agent

		offer = None

		if buy:
			pq = self.buy_offer
		else:
			pq = self.sell_offer

		if not pq.empty():
			# if not empty, check if highest buy/ lowest sell offer is from the agent
			offer_price, offer_agent = pq.get()

			if offer_agent != agent:
				offer = (offer_price, offer_agent)
			else:
				# if the offer comes from himself, then we need to look at the next offer
				offer = self.get_next_offer(agent, buy)

				# add the self offer to the list
				pq.put((offer_price, offer_agent))

		return offer


	def _take_action(self, action, agent):
		# action: nparray [buy, sell, buy_price, sell_price]
		# agent: the index of the agent who wants to take the action

		#print("agent: "+str(agent)+" action: ", str(action))
		if action[0] == 1:
			# buy
			buy_price = action[2]

			# settle offers
			# find lowest sell offer
			offer = self.get_next_offer(agent = agent, buy = False)
			if offer:
				lowest_sell, sell_agent = offer

				# if the offer is settled
				if lowest_sell <= buy_price:
					# the agent take the sell offer at the lowest sell price
					self.balance[agent] -= lowest_sell
					self.balance[sell_agent] += lowest_sell
					self.contract[agent] += 1
					self.contract[sell_agent] -= 1
				else:
					# add the offers to the offer queue (buy price is set to be negative)
					self.sell_offer.put(offer)
					self.buy_offer.put((-buy_price, agent))
			else:
				# add the offers to the offer queue (buy price is set to be negative)
				self.buy_offer.put((-buy_price, agent))

		if action[1] == 1:
			# sell
			sell_price = action[3]

			# settle offers
			# find lowest highest buy (buy price is set to be negative)
			offer = self.get_next_offer(agent = agent, buy = True)
			if offer:
				highest_buy, buy_agent = offer
				highest_buy = -highest_buy

				# if the offer is settled
				if highest_buy >= sell_price:
					# the agent take the buy offer at the highest buy price
					self.balance[agent] += highest_buy
					self.balance[buy_agent] -= highest_buy
					self.contract[agent] -= 1
					self.contract[buy_agent] += 1
				else:
					# add the offers to the offer queue (buy price is set to be negative)
					self.sell_offer.put((sell_price, agent))
					self.buy_offer.put(offer)
			else:
				# add the offers to the offer queue
				self.sell_offer.put((sell_price, agent))
		return

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		if self.day == self.number_of_sub_piles:
			print("\n\n====================== End Game ======================")
		elif self.sequence_counter == 1:
			print("\n\n====================== Beginning of day ", self.day, " ======================")
		else:
			print("=================== Before sequence ", self.sequence_counter, " ===================")

		print("balances: ", self.balance) #[our_agent, opponent1, opponent2, ...]
		print("contracts: ", self.contract)
		print("total reward: ", self.total_reward )


		if self.day == self.number_of_sub_piles:
			# print net worth at the end of game
			print("total net worth: ", self.balance + self.contract * self.public_pile.sum())
		else:
			print("turn sequence: ", self.turn_sequence)


if __name__ == "__main__":
	# Validate the environment

	from stable_baselines.common.env_checker import check_env
	env = TradingGameEnv() # 2 agents, 1 suit, 4 sub-piles (3days)

	# If the environment don't follow the interface, an error will be thrown
	check_env(env, warn=True)
	print("The environment is valid.")
