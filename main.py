import TradingGameEnv
import baseline_agents
from stable_baselines.common.env_checker import check_env

# Validate the environment

# add 1 baseline agent
agents = [baseline_agents.BaselineAgent1(), baseline_agents.BaselineAgent2()]
env = TradingGameEnv.TradingGameEnv(player_count = 3, other_agent_list = agents) # 2 agents, 1 suit, 4 sub-piles (3days)
# baseline1 takes action, then baseline2, then our agent

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
	obs, reward, done, info = env.step([1,1,40,45]) # buy at 40; sell at 45
	env.render()
	print('obs=', obs, 'reward=', reward, 'done=', done)
	if done:
		break

		

