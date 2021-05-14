import numpy as np
class BaselineAgent1():
	# TODO

	def __init__(self):
		...

	def predict(self, obs):
		#print("baseline1 action: ", [1,1,240,245])
		return [1,1,2000,3000] # buy at 240, sell at 250

class BaselineAgent2():
	# TODO

	def __init__(self):
		...

	def predict(self, obs):
		return [1,1,2200,2800] # buy at 400, sell at 10

class EVAgent():
	'''
	Agent calculates the E.V. of public pile and places bets accordingly
	'''
	def __init__(self, agent_idx, num_players, cards_per_suit, player_hand_count, public_cards_count, betting_margin = 100, suit_count = 1):

		self.num_players = num_players
		self.cards_per_suit = cards_per_suit
		self.SUIT_SUM = (self.cards_per_suit)*(1 + self.cards_per_suit)/2
		self.suit_count = suit_count
		self.player_hand_count = player_hand_count
		self.betting_margin = betting_margin
		self.public_cards_count = public_cards_count

	def predict(self, obs):

		'''
		    (total_sum - own_hand_sum - public_revealed_sum)
		EV = 		---------------------------
					  number_of_unknown_cards
		val = Revealed_public_pile_sum + (#unrevealed_cards_in_public_pile)*(EV_unrevealed_cards)
		'''
		EV = (self.SUIT_SUM - obs[-self.player_hand_count:].sum() - obs[:self.public_cards_count].sum())/(self.public_cards_count - np.count_nonzero(obs[:self.public_cards_count]) + (self.num_players - 1)*self.player_hand_count)
		val = obs[:self.public_cards_count].sum() + (self.public_cards_count - np.count_nonzero(obs[:self.public_cards_count]))*(EV)


		self.val = [1, 1, val - self.betting_margin, val + self.betting_margin]
		#print("EV Agent action", self.val)

		return self.val

class PlayerAgent():
	'''
	Agent that asks for human player to enter an action
	'''
	def __init__(self):
		...

	def predict(self, obs):

		print("Now it's your turn")
		print("Your observation: ", obs)
		print("Please enter your action. format: int int int int e.g. 1 1 33 33")
		
		action = [int(element) for element in input().split()]
		
		return action
