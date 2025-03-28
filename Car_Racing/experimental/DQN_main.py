# from DQN_agent import RL_Agent
# from DQN_env import gym_Env_Wrapper as gym_Wrapper
# import gym
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace

# gamma = 0.90
# batch_size = 128
# memory_size = 30000

# epsilon=1
# epsilon_decay = 650
# epsilon_min= 0.00

# skip_frames=4
# rescale_factor = 0.5
# mini_render =  True
# max_steps = 2200
# target_update = 1000

# # DEATH PARAMETERS
# death_parameters = {
#                     "death_memory_size": 8000,
#                     "death_batch_size": 4,
#                     "death_steps": 25,
#                     "death_tries": 1,
#                     "death_epsilon": 0.70 ## doesnt matter anymore
#                     }
# # death_memory_size = 16000
# # death_steps = 50
# # death_tries = 10
# # death_epsilon = 0.70


# smb= gym_super_mario_bros.make("SuperMarioBros-v0")
# smb = JoypadSpace(smb, SIMPLE_MOVEMENT)

# env = gym_Wrapper(smb, mini_render, skip_frames, rescale_factor, max_steps)

# agent = RL_Agent(env, memory_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, target_update, death_parameters)

# agent.train(10000) #5000 * 250 = 1,250,000 steps
# agent.test(5)


# # agent.load_model(agent.online_network,"bestDDQN.pt")
# # agent.test(10)

import argparse
import gym

from DQN_env import gym_Env_Wrapper as gym_Wrapper
from DQN_agent import RL_Agent as DQN_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train/test Super Mario agents.")
    
    parser.add_argument("--training_algorithm", type=str, default="Double_DQN",
                        choices=["DQN", "Double_DQN", "Dueling_DQN"],
                        help="Which agent class to use for training.")
    
    # Common agent parameters
    parser.add_argument("--gamma", type=float, default=0.85,
                        help="Discount factor.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size.")
    parser.add_argument("--memory_size", type=int, default=8192,
                        help="Replay memory size.")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial epsilon.")
    parser.add_argument("--epsilon_decay", type=float, default=1000,
                        help="Epsilon decay schedule.")
    parser.add_argument("--epsilon_min", type=float, default=0.00,
                        help="Minimum epsilon value.")
    parser.add_argument("--update_freq", type=int, default=1000,
                        help="Update frequency (used by some agents).")
    # parser.add_argument("--target_update", type=int, default=1000,
    #                     help="Target network update frequency (Double_DQN only).")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                        help="Learning rate.")
    parser.add_argument("--save_name", type=str, default="Double_DQN",
                        help="Prefix/name for saved models/logs.")
    
    # Environment parameters
    parser.add_argument("--skip_frames", type=int, default=4,
                        help="Number of frames to skip.")
    parser.add_argument("--mini_render", type=bool, default=True,
                        help="Enable mini rendering.")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Maximum steps per episode.")
    # parser.add_argument("--rescale_factor", type=float, default=0.5,
    #                     help="Rescale factor for the environment.")
    
    # Training settings
    parser.add_argument("--train_episodes", type=int, default=5000,
                        help="Number of training episodes.")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of test episodes after training.")
    
    # Death parameters (only used in experimental agents)
    parser.add_argument("--death_memory_size", type=int, default=1024,
                        help="Memory size for 'death' experiences.")
    parser.add_argument("--death_batch_size", type=int, default=64,
                        help="Batch size for 'death' experiences.")
    parser.add_argument("--death_steps", type=int, default=25,
                        help="Extra steps for training upon 'death'.")
    parser.add_argument("--death_tries", type=int, default=1,
                        help="Number of attempts in 'death' scenario.")
    parser.add_argument("--death_start_weights", 
                        type=lambda s: list(map(float, s.split(','))),
                        default=[0.4, 0.4, 0.2],
                        help="Comma-separated starting weights for death (e.g., '0.4,0.4,0.2').")
    parser.add_argument("--death_end_weights", 
                        type=lambda s: list(map(float, s.split(','))),
                        default=[0.7, 0.2, 0.1],
                        help="Comma-separated ending weights for death (e.g., '0.7,0.2,0.1').")
    parser.add_argument("--death_weights_ep_transition", type=int, default=1000,
                        help="Episode transition for death weight interpolation.")
    
    parser.add_argument("--continue_training", type=str, default=None, help="Continue training a model found inside the 'TrainModelResults' folder, by this name")
    
    parser.add_argument("--test_agent", type=str, default=None, help="Test a model found inside the 'TrainModelResults' folder, by this name")
    parser.add_argument("--test_epsilon", type=float, default=0, help="Random actions in test")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build agent parameters dictionary
    agent_params = {
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "update_freq": args.update_freq,
        # "target_update": args.target_update,
        "learning_rate": args.learning_rate,
        "save_name": args.save_name,
        "training_algorithm": args.training_algorithm
    }
    
    # Build environment parameters dictionary
    env_params = {
        "skip_frames": args.skip_frames,
        "mini_render": args.mini_render,
        "max_steps": args.max_steps,
        "env_name": args.save_name
        # "rescale_factor": args.rescale_factor
    }
    
    death_params = {
        # Death parameters (agents that don't use them can ignore these)
        "death_memory_size": args.death_memory_size,
        "death_batch_size": args.death_batch_size,
        "death_steps": args.death_steps,
        "death_tries": args.death_tries,
        "death_start_weights": args.death_start_weights,
        "death_end_weights": args.death_end_weights,
        "death_weights_ep_transition": args.death_weights_ep_transition
    }
    
    car_racer = gym.make('CarRacing-v2', domain_randomize=True, continuous=False)
    env = gym_Wrapper(car_racer, env_params)
    
    agent_class = DQN_agent
    agent = agent_class(env, agent_params,death_params)
    
    if(args.test_agent is None):
        if(args.continue_training is not None):
            agent.load_model(args.continue_training)
            print("Loaded: ", args.continue_training)
        agent.train(args.train_episodes)
        agent.test(args.test_episodes)
    else:
        agent.load_model(args.test_agent)
        agent.epsilon = args.test_epsilon
        agent.test(args.test_episodes)

if __name__ == "__main__":
    main()
