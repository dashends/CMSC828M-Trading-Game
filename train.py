import TradingGameEnv
import baseline_agents
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy


# add 1 baseline agent
agents = [baseline_agents.EVAgent()]
env = TradingGameEnv.TradingGameEnv(player_count = 2, other_agent_list = agents, seq_per_day = 30, cards_per_suit = 100, random_seq = True)
# 2 agents, 1 suit, 4 sub-piles (3days), 30 days, randomized sequence for each day

model = PPO2('MlpPolicy', env)
model.learn(total_timesteps = 100000)
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


env.render()
print('obs=', obs)

while(True):
	print('Taking actions for the current sequence......')
	action, _states = model.predict(obs)
	print("action: ", action)
	obs, reward, done, info = env.step(action)
	print('reward=', reward, 'done=', done)
	print("total net worth: ", env.balance + env.contract * env.public_pile.sum())
	env.render()
	print('obs=', obs)
	if done:
		break
