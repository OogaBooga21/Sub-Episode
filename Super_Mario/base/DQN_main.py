# from DQN_agent import RL_Agent as DQN_agent
# from Double_DQN_agent import RL_Agent as Double_DQN_agent
# from Dueling_DQN_agent import RL_Agent as Dueling_DQN_agent
# from DQN_env import gym_Env_Wrapper as gym_Wrapper
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace
# import gym

# # gamma = 0.99
# # batch_size = 12
# # memory_size = 192

# # epsilon=1
# # epsilon_decay = 7000
# # epsilon_min= 0.05

# # skip_frames=4
# # rescale_factor = 0.5
# # mini_render =  False
# # max_steps = 90000
# # update_freq = 1000
# save_name = "Double_DQN"

# agent_params = {
#     "gamma": 0.99,
#     "batch_size": 8,
#     "memory_size": 10,
#     "epsilon": 1,
#     "epsilon_decay": 7000,
#     "epsilon_min": 0.02,
#     "update_freq": 1000,
#     "learning_rate": 0.00025,
#     "save_name": save_name
# }
# env_params = {
#     "skip_frames": 4, #standard, dont rly change
#     "mini_render": True, 
#     "max_steps": 90000
# }


# agent_train = "Double_DQN"

# smb= gym_super_mario_bros.make("SuperMarioBros-v0")
# smb = JoypadSpace(smb, SIMPLE_MOVEMENT)
# env = gym_Wrapper(smb, env_params)

# if agent_train == "DQN":
#     agent = DQN_agent(env, agent_params)
# elif agent_train == "Double_DQN":
#     agent = Double_DQN_agent(env, agent_params)
# elif agent_train == "Dueling_DQN":
#     agent = Dueling_DQN_agent(env, agent_params)

# agent.train(15000) #5000 * 250 = 1,250,000 steps
# agent.test(5)


# # agent.load_model("bestDQN.pt")
# # agent.test(10)




# DQN_main.py

import argparse
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from DQN_env import gym_Env_Wrapper as gym_Wrapper
from DQN_agent import RL_Agent as DQN_agent
# from DQN_agent import RL_Agent as Double_DQN_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train/test Super Mario agents.")
    
    # Which agent are we training? (DQN, Double_DQN, Dueling_DQN)
    parser.add_argument("--training_algorithm", type=str,
                        choices=["DQN", "Double_DQN", "Dueling_DQN"],
                        help="Which agent class to use for training.")
    
    # Common agent parameters
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--memory_size", type=int, default=30000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=500)
    parser.add_argument("--epsilon_min", type=float, default=0.00)
    parser.add_argument("--update_freq", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--save_name", type=str, default="Double_DQN",
                        help="Prefix/name for saved models/logs.")
    
    # Environment parameters
    parser.add_argument("--skip_frames", type=int, default=4)
    parser.add_argument("--mini_render", type=bool, default=True)
    parser.add_argument("--max_steps", type=int, default=90000)
    
    # Training settings
    parser.add_argument("--train_episodes", type=int, default=10000,
                        help="Number of episodes to train.")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of episodes to test after training.")
    
    parser.add_argument("--continue_training", type=str, default=None, help="Continue training a model found inside the 'TrainModelResults' folder, by this name")
    
    parser.add_argument("--test_agent", type=str, default=None, help="Test a model found inside the 'TrainModelResults' folder, by this name")
    parser.add_argument("--test_epsilon", type=float, default=0, help="Random actions in test")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build agent_params from the arguments
    agent_params = {
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "update_freq": args.update_freq,
        "learning_rate": args.learning_rate,
        "save_name": args.save_name,
        "training_algorithm": args.training_algorithm
    }
    
    # Build environment params
    env_params = {
        "skip_frames": args.skip_frames,
        "mini_render": args.mini_render,
        "max_steps": args.max_steps,
        "env_name": args.save_name
    }
    
    # Create the Super Mario environment
    smb = gym_super_mario_bros.make("SuperMarioBros-v0")
    smb = JoypadSpace(smb, SIMPLE_MOVEMENT)
    env = gym_Wrapper(smb, env_params)
    
    # Select the agent class based on --agent_train
    # if args.training_algorithm == "DQN":
    #     agent_class = DQN_agent
    # elif args.training_algorithm == "Double_DQN":
    #     agent_class = Double_DQN_agent
    # elif args.training_algorithm == "Dueling_DQN":  # "Dueling_DQN"
    #     agent_class = Dueling_DQN_agent
    
    agent_class = DQN_agent
    agent = agent_class(env, agent_params)
    
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
