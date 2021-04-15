import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np

NUM_PLAYERS = 2
SEQ_PER_DAY = 30
CARDS_PER_SUIT = 100
SUIT_COUNT = 1

hand_count = (int) ((CARDS_PER_SUIT)/2*SUIT_COUNT/NUM_PLAYERS)
SUIT_LIST = np.arange(1, CARDS_PER_SUIT+1)
np.random.shuffle(SUIT_LIST)


HANDS = SUIT_LIST[0:NUM_PLAYERS * hand_count].reshape((NUM_PLAYERS, hand_count))
PUBLIC_PILE = SUIT_LIST[NUM_PLAYERS * hand_count:]


# add 1 baseline agent
agents = [baseline_agents.EVAgent(agent_idx = 0, num_players = NUM_PLAYERS, cards_per_suit = CARDS_PER_SUIT, hands = HANDS, player_hand_count = hand_count, public_pile = PUBLIC_PILE)]
env = TradingGameEnv.TradingGameEnv(suit_list = SUIT_LIST, player_count = NUM_PLAYERS, other_agent_list = agents, seq_per_day = SEQ_PER_DAY, cards_per_suit = CARDS_PER_SUIT, random_seq = True)
# 2 agents, 1 suit, 4 sub-piles (3days), 30 days, randomized sequence for each day

model = PPO2('MlpPolicy', env)
model.learn(total_timesteps = 10000)
#model.save("test_model_1")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean_reward: ", mean_reward)
print("std_reward: ", std_reward)

# Playing test rounds
print("Playing test rounds:")
obs = env.reset()

print("obs space", env.observation_space)
print("action space", env.action_space)
print("sample action", env.action_space.sample())
print("public pile", env.public_pile)
print("hands", env.hands)
print("Groud Truth Value of Future's", env.public_pile.sum())


print("suit sum ", agents[0].SUIT_SUM)
print("hand's sum ", agents[0].hand_sum)
print("cards per suit ", agents[0].cards_per_suit)
print("player's cards ", agents[0].player_hand_count)
print("betting range ", agents[0].val)

env.render()
print('obs=', obs)

while(True):
	print('Taking actions for the current sequence......')
	action, _states = model.predict(obs)
	#agent_predict = agents[0].predict()
	print("action: ", action)
	#print("agent's action ", agent_predict)
	obs, reward, done, info = env.step(action)
	print('reward=', reward, 'done=', done)
	print("total net worth: ", env.balance + env.contract * env.public_pile.sum())
	env.render()
	print('obs=', obs)
	if done:
		break
