import gym
from gym import spaces
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import math
from queue import PriorityQueue
 
INITIAL_ACCOUNT_BALANCE = 10000
SUIT_SUM = (1+13)*13/2
AGENT_INDEX = 0
	
class TradingGameEnv(gym.Env):
	"""A trading game environment for OpenAI gym"""
	metadata = {'render.modes': ['human']}

	
	def __init__(self, player_count = 2, suit_count = 1, number_of_sub_piles = 4, other_agent_list = []):
		super(TradingGameEnv, self).__init__()
		self.player_count = player_count
		self.suit_count = suit_count
		self.number_of_sub_piles = number_of_sub_piles
		self.other_agents = other_agent_list
		
		# Actions: buy (1=true), sell, price for buy, price for buy
		self.action_space =  spaces.MultiDiscrete([2, 2, SUIT_SUM, SUIT_SUM])
		
		# 2-6 player, about 6 cards are reserved for the public pile
		# each player is given (13-6)*suit_count/player_count private cards
		self.player_hand_count = (int) ((13-6)*self.suit_count/player_count)
		
		self.public_cards_count = 13- self.player_hand_count*player_count
		
		# sub-piles
		self.public_sub_pile_cards_count = math.ceil((13 - self.public_cards_count)/self.number_of_sub_piles)
		
		# observation: revealed public pile + own hand  + balance + contract
		"""
		dict_space = {
			'public_pile': spaces.Box(low=0, high=13, shape=(4, self.public_sub_pile_cards_count), dtype=np.int),
			'own_hand': spaces.Box(low=0, high=13, shape=(self.player_hand_count,), dtype=np.int)
			#'balance': spaces.Box(low=-np.inf, high=np.inf, shape=(self.player_count,), dtype=np.float32),
			#'contract': spaces.Box(low=-np.inf, high=np.inf, shape=(self.player_count,), dtype=np.float32)
		}
		self.observation_space = spaces.Dict(dict_space)
		"""
		# first public_cards_count cards are public pile
		# then self.player_hand_count is own_hand
		self.obs_element_count = self.public_cards_count + self.player_hand_count
		lower = np.zeros(self.obs_element_count, dtype=np.int)
		upper = np.full(self.obs_element_count, 13, dtype=np.int)
		self.observation_space = spaces.Box(lower , upper, dtype=np.int)

	
	def reset(self):
		# Reset the state of the environment to an initial state
		# the agent is place at AGENT_INDEX = 0 row
		self.balance = np.full(self.player_count, INITIAL_ACCOUNT_BALANCE, dtype=np.int)
		self.contract = np.zeros(self.player_count, dtype=np.int)
		self.time_step = 1
		self.total_reward = 0
		
		# deal cards
		suit_list = np.arange(1, 14)
		np.random.shuffle(suit_list)

		self.hands = suit_list[0:self.player_count * self.player_hand_count].reshape((self.player_count, self.player_hand_count))
		
		
		self.public_pile = suit_list[self.player_count * self.player_hand_count:]
		
			
		self.sell_offer = PriorityQueue(self.player_count)
		self.buy_offer = PriorityQueue(self.player_count)
		
		
		return self._next_observation()
  
	def step(self, action):
		# let other baseline agents to take actions
		if len(self.other_agents) == self.player_count-1:
			for i in range(self.player_count-1):
				obs_i = self._next_observation(i+1)
				action_i = self.other_agents[i].predict(obs_i)
				self._take_action(action_i, i+1)
		
		# Execute one time step within the environment
		self._take_action(action, AGENT_INDEX)
		
		
		# compute reward
		# reward = expected profit * timestep
		#  = (expected value of public pile * amount of contract + balance 
		# 		– initial balance ) * timestep
		#  expected value of public pile = 
		#     (revealed public pile + expected un-revealed public card value * un-revealed card count)
		revealed_card_index = min(self.time_step*self.public_sub_pile_cards_count, self.public_pile.shape[0]) 
		revealed_value = self.public_pile[0:revealed_card_index].sum()
		
		# expected un-revealed public card value = (sum of all cards 
		# 			– revealed public piles – own hand)/(remaining card count)
		expected_card_value = ((SUIT_SUM - revealed_value - self.hands[AGENT_INDEX,:].sum()) /
							(13 - self.player_hand_count - (revealed_card_index)))
		expected_value_of_public_pile = (revealed_value + expected_card_value  * 
							(self.public_pile.shape[0] - self.time_step*self.public_sub_pile_cards_count))
		
		reward = (expected_value_of_public_pile * self.contract[AGENT_INDEX] + 
							self.balance[AGENT_INDEX] - INITIAL_ACCOUNT_BALANCE)
		reward *= self.time_step
		
		
		# if it's the last day, give final reward
		# 
		if self.time_step == self.number_of_sub_piles-1:
			reward += ((self.public_pile.sum() * self.contract[AGENT_INDEX] + 
							self.balance[AGENT_INDEX] - INITIAL_ACCOUNT_BALANCE)*
							(self.time_step + 1))
		
		self.total_reward += reward
		
		self.time_step += 1
		done = self.time_step >= self.number_of_sub_piles
		obs = self._next_observation()
		
		# clear the offers for the day
		with self.sell_offer.mutex:
			self.sell_offer.queue.clear()
		with self.buy_offer.mutex:
			self.buy_offer.queue.clear()
			
		return obs, reward, done, {}
		
	def _next_observation(self, agent_index = AGENT_INDEX):
		obs = np.zeros(self.obs_element_count, dtype=np.int)
		
		# public pile
		revealed_card_index = min(self.time_step*self.public_sub_pile_cards_count, self.public_pile.shape[0])
		obs[0:revealed_card_index] = self.public_pile[0:revealed_card_index]
		
		# own hand
		obs[self.public_cards_count:] = self.hands[agent_index, :]
		
		return obs
		
	def _take_action(self, action, agent):
		if action[0] == 1:
			# buy
			buy_price = action[2]
			
			# settle offers
			# find lowest sell
			if not self.sell_offer.empty():
				lowest_sell, sell_agent = self.sell_offer.get()
				
				# if the offer is settled
				if lowest_sell <= buy_price:
					# the agent take the sell offer at the lowest sell price
					self.balance[agent] -= lowest_sell
					self.balance[sell_agent] += lowest_sell
					self.contract[agent] += 1
					self.contract[sell_agent] -= 1
				else:
					# add the offers to the offer queue (buy price is set to be negative)
					self.sell_offer.put((lowest_sell, sell_agent))
					self.buy_offer.put((-buy_price, agent))
			else:
				self.buy_offer.put((-buy_price, agent))
				
		if action[1] == 1:
			# sell
			sell_price = action[3]
			
			# settle offers
			# find lowest highest buy (buy price is set to be negative)
			if not self.buy_offer.empty():
				highest_buy, buy_agent = self.buy_offer.get()
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
					self.buy_offer.put((-highest_buy, buy_agent))
			else:
				self.sell_offer.put((sell_price, agent))
				
		return
	
	def render(self, mode='human', close=False):
		# Render the environment to the screen
		if self.time_step == self.number_of_sub_piles:
			print("************************ End Game ************************")
		else:
			print("******************** Beginning of day ", self.time_step, "********************")
		
		print("balances: ", self.balance)
		print("contracts: ", self.contract)
		print("total reward: ", self.total_reward )
		
		
if __name__ == "__main__":
	# Validate the environment
	
	from stable_baselines.common.env_checker import check_env
	env = TradingGameEnv() # 2 agents, 1 suit, 4 sub-piles (3days)
	
	# If the environment don't follow the interface, an error will be thrown
	check_env(env, warn=True)
	print("The environment is valid.")