import gym
from gym import spaces
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import math
from queue import PriorityQueue

INITIAL_ACCOUNT_BALANCE = 10000
AGENT_INDEX = 0
NO_ACTION_PENALTY = -5

class TradingGameEnv(gym.Env):
	"""A trading game environment for OpenAI gym"""
	metadata = {'render.modes': ['human']}


	def __init__(self, player_count = 2, suit_count = 1, number_of_sub_piles = 4, other_agent_list = [], seq_per_day = 1, random_seq = False, cards_per_suit = 13):
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

		if len(self.other_agents) != self.player_count-1:
			print("Error: other_agent_list do not conform to the number of players. You may need to add/remove some other agents")

		# Actions: buy (1=true), sell, price for buy, price for sell
		# action: [1, 0, 23, 25]
		self.action_space =  spaces.MultiDiscrete([2, 2, self.SUIT_SUM, self.SUIT_SUM])

		# about half cards are reserved for the public pile
		# each player is given player_hand_count private cards
		self.player_hand_count = (int) ((self.cards_per_suit)/2*self.suit_count/player_count)

		self.public_cards_count = self.cards_per_suit*self.suit_count - self.player_hand_count*player_count

		# sub-piles
		self.public_sub_pile_cards_count = math.ceil(self.public_cards_count/self.number_of_sub_piles)

		# observation: revealed public pile + own hand  + balance + contract

		# first public_cards_count cards are public pile
		# then self.player_hand_count is own_hand
		# [public pile + own hand]
		# 6 cards for public pile, 3 cards in own hand
		# [1, 5, 0, 0, 0, 0, 2, 11, 12]
		#Last numbers are own hand.
		self.obs_element_count = self.public_cards_count + self.player_hand_count
		lower = np.zeros(self.obs_element_count, dtype=np.int) #[0 0 0 0 0]
		upper = np.full(self.obs_element_count, self.cards_per_suit, dtype=np.int)
		self.observation_space = spaces.Box(lower , upper, dtype=np.int)
		### TODO: Modify size of obs space.



	def reset(self):
		# Reset the state of the environment to an initial state
		# the agent is place at AGENT_INDEX = 0 row
		self.balance = np.full(self.player_count, INITIAL_ACCOUNT_BALANCE, dtype=np.int) #[agent, other_agent1, other_agent2, ....]
		self.contract = np.zeros(self.player_count, dtype=np.int)
		self.sequence_counter = 1
		self.day = 1
		self.total_reward = 0
		if self.random_seq:
			np.random.shuffle(self.turn_sequence)

		# deal cards
		suit_list = np.arange(1, self.cards_per_suit+1)
		np.random.shuffle(suit_list)

		self.hands = suit_list[0:self.player_count * self.player_hand_count].reshape((self.player_count, self.player_hand_count))


		self.public_pile = suit_list[self.player_count * self.player_hand_count:]


		self.sell_offer = PriorityQueue()
		self.buy_offer = PriorityQueue()


		return self._next_observation()

	def step(self, action):
		### TODO: Use data structures to recor transaction history
		##

		# take actions based upon turn sequence
		for i in self.turn_sequence:

			# if it is an opponent agent
			if i != AGENT_INDEX:
				obs_i = self._next_observation(i)
				action_i = self.other_agents[i-1].predict(obs_i)
				self._take_action(action_i, i)

			else:
				# if it is our agent

				# Execute one time step within the environment
				self._take_action(action, AGENT_INDEX)

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
		# add some panelty if the agent is doing nothing
		if (action[0] == 0 and action[1] == 0):
			reward += NO_ACTION_PENALTY
		reward *= self.day

		# if it's the last day, give final reward
		if self.day == self.number_of_sub_piles-1 and self.sequence_counter == self.seq_per_day:
			final_reward = ((self.public_pile.sum() * self.contract[AGENT_INDEX] +
							self.balance[AGENT_INDEX] - INITIAL_ACCOUNT_BALANCE)*
							(self.day + 1))
			reward += final_reward

		self.total_reward += reward

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



		return obs, reward, done, {}

	def _next_observation(self, agent_index = AGENT_INDEX):
		# get the observation for the agent at the index
		## TODO: Populate observation array.
		obs = np.zeros(self.obs_element_count, dtype=np.int)

		# public pile
		revealed_card_index = min(self.day*self.public_sub_pile_cards_count, self.public_pile.shape[0])
		obs[0:revealed_card_index] = self.public_pile[0:revealed_card_index]

		# own hand
		obs[self.public_cards_count:] = self.hands[agent_index, :]

		# return [1, 5, 0, 0, 0, 0, 2, 11, 12]
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
