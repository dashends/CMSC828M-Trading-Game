import TradingGameEnv
import baseline_agents
from stable_baselines.common.env_checker import check_env

# Validate the environment

# add 1 baseline agent
agents = [baseline_agents.BaselineAgent1(), baseline_agents.BaselineAgent2()]
env = TradingGameEnv.TradingGameEnv(player_count = 3, other_agent_list = agents, seq_per_day = 3, cards_per_suit = 30, random_seq = True) 
# 2 agents, 1 suit, 4 sub-piles (3days), 30 days, randomized sequence for each day

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)
print("The environment is valid.")

# Playing test rounds
print("Playing test rounds:")
obs = env.reset()

print("obs space", env.observation_space)
print("action space", env.action_space)
print("sample action", env.action_space.sample())
print("public pile", env.public_pile)
print("hands", env.hands)

env.render()
print('obs=', obs)

while(True):
	print('Taking actions for the current sequence......')
	obs, reward, done, info = env.step([1,1, 230, 240]) # buy at 230; sell at 240
	print('reward=', reward, 'done=', done)
	env.render()
	print('obs=', obs)
	if done:
		break
	