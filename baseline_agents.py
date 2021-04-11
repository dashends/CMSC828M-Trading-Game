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
	def __init__(self, obs, agent_idx, betting_margin = 100):
		self.hand = obs.hands[agent_idx]
		self.hand_sum = self.hand.sum()

	def predict(self, obs):

		'''
		E_GroundTruth = totalSum
		- hand_sum
		- public_revealed_pile_sum
		- #opponents*#cards*avgCardVal
		'''
		avg_card_val = (obs.SUIT_SUM*obs.suit_count-self.hand_sum)/(obs.cards_per_suit - obs.player_hand_count)
		EV = obs.SUIT_SUM*obs.suit_count \
			 - self.hand_sum  \
			 - obs.observation_space[:obs.public_cards_count].sum() \
			 - (obs.player_count-1)*(obs.player_hand_count)*(avg_card_val)

		return [1, 1, EV-betting_margin, EV+betting_margin]
