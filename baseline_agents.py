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
		#print("baseline2 action:", [1,1,400,10])
		return [1,1,2200,2800] # buy at 400, sell at 10

class EVAgent():
	'''
	Agent calculates the E.V. of public pile and places bets accordingly
	'''
	def __init__(self, agent_idx, num_players, cards_per_suit, hands, player_hand_count, public_pile, betting_margin = 100, suit_count = 1):
		self.hand = hands[agent_idx]
		self.hand_sum = self.hand.sum()
		self.num_players = num_players
		self.cards_per_suit = cards_per_suit
		self.hand_sum = self.hand.sum()
		self.SUIT_SUM = (self.cards_per_suit)*(1 + self.cards_per_suit)/2
		self.suit_count = suit_count
		self.player_hand_count = player_hand_count
		self.public_pile = public_pile
		self.betting_margin = betting_margin

	def predict(self, env):

		'''
		E_GroundTruth = totalSum
		- hand_sum
		- public_revealed_pile_sum
		- #opponents*#cards*avgCardVal
		'''
		avg_card_val = (self.SUIT_SUM * self.suit_count - self.hand_sum)/(self.cards_per_suit - self.player_hand_count)
		EV = self.SUIT_SUM*self.suit_count \
			 - self.hand_sum  \
			 - (self.num_players-1)*(self.player_hand_count)*(avg_card_val)
		self.val = [1, 1, EV - self.betting_margin, EV + self.betting_margin]

		return self.val
