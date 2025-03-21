from DQN_agent import RL_Agent
from DQN_env import gym_Env_Wrapper as gym_Wrapper
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

gamma = 0.90
batch_size = 128
memory_size = 30000

epsilon=1
epsilon_decay = 650
epsilon_min= 0.00

skip_frames=4
rescale_factor = 0.5
mini_render =  True
max_steps = 2200
target_update = 1000

# DEATH PARAMETERS
death_parameters = {
                    "death_memory_size": 8000,
                    "death_batch_size": 4,
                    "death_steps": 25,
                    "death_tries": 1,
                    "death_epsilon": 0.70 ## doesnt matter anymore
                    }
# death_memory_size = 16000
# death_steps = 50
# death_tries = 10
# death_epsilon = 0.70


smb= gym_super_mario_bros.make("SuperMarioBros-v0")
smb = JoypadSpace(smb, SIMPLE_MOVEMENT)

env = gym_Wrapper(smb, mini_render, skip_frames, rescale_factor, max_steps)

agent = RL_Agent(env, memory_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, target_update, death_parameters)

agent.train(10000) #5000 * 250 = 1,250,000 steps
agent.test(5)


# agent.load_model(agent.online_network,"bestDDQN.pt")
# agent.test(10)