from DQN_agent import RL_Agent
from DQN_env import gym_Env_Wrapper as gym_Wrapper
import gym

gamma = 0.85
batch_size = 512
memory_size = 8192

epsilon=1
epsilon_decay = 2500
epsilon_min= 0.05

skip_frames=4
# target_update_freq = 1000  # Update target network every N steps
rescale_factor = 1
mini_render =  True
max_steps = 250
target_update = 1000

car_racer = gym.make('CarRacing-v2', domain_randomize=True, continuous=False)

env = gym_Wrapper(car_racer, mini_render, skip_frames, rescale_factor, max_steps)

agent = RL_Agent(env, memory_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, target_update)

agent.train(4000) #5000 * 250 = 1,250,000 steps
agent.test(5)


# agent.load_model(agent.online_network,"best.pt")
# agent.test(10)