import TradingGameEnv
import baseline_agents
from stable_baselines.common.env_checker import check_env

# Validate the environment

# add 1 baseline agent
agents = [baseline_agents.BaselineAgent1()]
env = TradingGameEnv.TradingGameEnv(player_count = 2, other_agent_list = agents, seq_per_day = 3, cards_per_suit = 30) 
# 2 agents, 1 suit, 4 sub-piles (3days), 30 days
# baseline1 takes action, then our agent

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
	obs, reward, done, info = env.step([1,1,230,240]) # buy at 230; sell at 240
	print('Taking actions for the current sequence...... \nreward=', reward, 'done=', done)
	env.render()
	print('obs=', obs)
	if done:
		break

		

