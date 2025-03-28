#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master entry point for RL experiments with conditional arguments."
    )
    # Required selection parameters
    parser.add_argument("--game", type=str, required=True,
                        choices=["Car_Racing", "Super_Mario"],
                        help="Which game folder to use.")
    parser.add_argument("--version", type=str, required=True,
                        choices=["base", "experimental"],
                        help="Which version folder to use (base or experimental).")
    parser.add_argument("--training_algorithm", type=str, required=True,
                        choices=["DQN", "Double_DQN", "Dueling_DQN"],
                        help="Which training algorithm to use.")

    # Common training parameters
    parser.add_argument("--train_episodes", type=int, default=None,
                        help="Number of training episodes.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size.")
    parser.add_argument("--memory_size", type=int, default=None,
                        help="Replay memory size.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Initial epsilon.")
    parser.add_argument("--epsilon_decay", type=int, default=None,
                        help="Epsilon decay schedule.")
    parser.add_argument("--epsilon_min", type=float, default=None,
                        help="Minimum epsilon value.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate.")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Name for saving models/logs.")

    # Environment parameters
    parser.add_argument("--skip_frames", type=int, default=None,
                        help="Number of frames to skip.")
    parser.add_argument("--mini_render", type=bool, default=True,
                        help="Enable mini rendering.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps in an episode.")

    # Double_DQN only
    parser.add_argument("--update_freq", type=int, default=None,
                        help="Target network update frequency (Double_DQN only).")

    # Death parameters (only used in experimental version)
    parser.add_argument("--death_memory_size", type=int, default=None,
                        help="Memory size for 'death' experiences.")
    parser.add_argument("--death_batch_size", type=int, default=None,
                        help="Batch size for 'death' experiences.")
    parser.add_argument("--death_steps", type=int, default=None,
                        help="Extra steps for training upon 'death'.")
    parser.add_argument("--death_tries", type=int, default=None,
                        help="Number of alternative attempts in 'death' scenario.")
    parser.add_argument("--death_start_weights",
                        type=lambda s: list(map(float, s.split(','))),
                        default="0.4,0.4,0.2",
                        help="Comma-separated starting weights for death (e.g., '0.4,0.4,0.2').")
    parser.add_argument("--death_end_weights",
                        type=lambda s: list(map(float, s.split(','))),
                        default="0.7,0.2,0.1",
                        help="Comma-separated ending weights for death (e.g., '0.7,0.2,0.1').")
    parser.add_argument("--death_weights_ep_transition", type=int, default=None,
                        help="Episode transition for death weight interpolation.")
    
    parser.add_argument("--continue_training", type=str, default=None, help="Continue training a model found inside the 'TrainModelResults' folder, by this name")
    parser.add_argument("--test_agent", type=str, default=None, help="Test a model found inside the 'TrainModelResults' folder, by this name")
    parser.add_argument("--test_epsilon", type=float, default=None, help="Random actions in test")
    
    parser.add_argument("--test_episodes", type=int, default=None,
                        help="Number of episodes to test.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    script_dir = os.path.join(os.path.dirname(__file__), args.game, args.version)
    script_path = os.path.join(script_dir, "DQN_main.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # Build the base command list
    command = [
        sys.executable,  # Uses the current Python interpreter
        script_path,
        "--training_algorithm",str(args.training_algorithm)
        # "--train_episodes", str(args.train_episodes),
        # "--gamma", str(args.gamma),
        # "--batch_size", str(args.batch_size),
        # "--memory_size", str(args.memory_size),
        # "--epsilon", str(args.epsilon),
        # "--epsilon_decay", str(args.epsilon_decay),
        # "--epsilon_min", str(args.epsilon_min),
        # "--learning_rate", str(args.learning_rate),
        # "--save_name", args.save_name,
        # "--skip_frames", str(args.skip_frames),
        # "--max_steps", str(args.max_steps),
        # # "--test_episodes", str(args.test_episodes),
        # "--training_algorithm", args.training_algorithm
    ]
    
    # if args.mini_render:
    #     command.append("--mini_render")
    
    if args.train_episodes is not None:
        command.extend(["--train_episodes", str(args.train_episodes)])
    if args.gamma is not None:
        command.extend(["--gamma", str(args.gamma)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.memory_size is not None:
        command.extend(["--memory_size", str(args.memory_size)])
    if args.epsilon is not None:
        command.extend(["--epsilon", str(args.epsilon)])
    if args.epsilon_decay is not None:
        command.extend(["--epsilon_decay", str(args.epsilon_decay)])
    if args.epsilon_min is not None:
        command.extend(["--epsilon_min", str(args.epsilon_min)])
    if args.learning_rate is not None:
        command.extend(["--learning_rate", str(args.learning_rate)])
    
    if args.skip_frames is not None:
        command.extend(["--skip_frames", str(args.skip_frames)])
    if args.max_steps is not None:
        command.extend(["--max_steps", str(args.max_steps)])
    if args.mini_render:
        command.extend(["--mini_render", str(args.mini_render)])
    
    if (args.training_algorithm == "Double_DQN") and (args.update_freq is not None):
        command.extend(["--update_freq", str(args.update_freq)])
    
    if args.version == "experimental":
        if args.death_memory_size is not None:
            command.extend(["--death_memory_size", str(args.death_memory_size)])
        if args.death_batch_size is not None:
            command.extend(["--death_batch_size", str(args.death_batch_size)])
        if args.death_steps is not None:
            command.extend(["--death_steps", str(args.death_steps)])
        if args.death_tries is not None:
            command.extend(["--death_tries", str(args.death_tries)])
        if args.death_start_weights is not None:
            command.extend(["--death_start_weights", ",".join(map(str, args.death_start_weights))])
        if args.death_end_weights is not None:
            command.extend(["--death_end_weights", ",".join(map(str, args.death_end_weights))])
        if args.death_weights_ep_transition is not None:
            command.extend(["--death_weights_ep_transition", str(args.death_weights_ep_transition)])
    
    if args.continue_training is not None:
        model_path = os.path.join("TrainModelResults", f"{args.continue_training}")
        command.extend(["--continue_training", str(model_path)])
        
    if args.test_agent is not None:
        model_path = os.path.join("TrainModelResults", f"{args.test_agent}")
        command.extend(["--test_agent", str(model_path)])
        
    if args.test_episodes is not None:
        command.extend(["--test_episodes", str(args.test_episodes)])
        
    if args.test_epsilon is not None:
        command.extend(["--test_epsilon", str(args.test_epsilon)])
    
    if args.save_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_name = f"{args.game}_{args.version}_{args.training_algorithm}_{ts}"
    
    command.extend(["--save_name", args.save_name])
    
    print("Running command:")
    print(" ".join(command))
    
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
