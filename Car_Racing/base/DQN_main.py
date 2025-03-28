# from DQN_agent import RL_Agent
# from DQN_env import gym_Env_Wrapper as gym_Wrapper
# import gym

# gamma = 0.85
# batch_size = 512
# memory_size = 8192

# epsilon=1
# epsilon_decay = 2500
# epsilon_min= 0.05

# skip_frames=4
# # target_update_freq = 1000  # Update target network every N steps
# rescale_factor = 1
# mini_render =  True
# max_steps = 250

# car_racer = gym.make('CarRacing-v2', domain_randomize=True, continuous=False)

# env = gym_Wrapper(car_racer, mini_render, skip_frames, rescale_factor, max_steps)

# agent = RL_Agent(env, memory_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size)

# agent.train(4000) #5000 * 250 = 1,250,000 steps
# agent.test(5)


# # agent.load_model(agent.network,"C:/Users/Oli/Documents/GitHub/Sub-Episode/best.pt")
# # agent.test(10)

import argparse
import gym

from DQN_env import gym_Env_Wrapper as gym_Wrapper
from DQN_agent import RL_Agent as DQN_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train/test Car Racing agents.")
    
    # Which agent are we training? (DQN, Double_DQN, Dueling_DQN)
    parser.add_argument("--training_algorithm", type=str,
                        choices=["DQN", "Double_DQN", "Dueling_DQN"],
                        help="Which agent class to use for training.")
    
    # Common agent parameters
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--memory_size", type=int, default=8192)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=1000)
    parser.add_argument("--epsilon_min", type=float, default=0.00)
    parser.add_argument("--update_freq", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--save_name", type=str, default="Double_DQN",
                        help="Prefix/name for saved models/logs.")
    
    # Environment parameters
    parser.add_argument("--skip_frames", type=int, default=4)
    parser.add_argument("--mini_render", type=bool, default=True)
    parser.add_argument("--max_steps", type=int, default=250)
    
    # Training settings
    parser.add_argument("--train_episodes", type=int, default=4000,
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
    
    car_racer = gym.make('CarRacing-v2', domain_randomize=True, continuous=False)
    env = gym_Wrapper(car_racer, env_params)
    
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