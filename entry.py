#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

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
    parser.add_argument("--train_episodes", type=int, default=5000,
                        help="Number of training episodes.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--memory_size", type=int, default=10,
                        help="Replay memory size.")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial epsilon.")
    parser.add_argument("--epsilon_decay", type=int, default=7000,
                        help="Epsilon decay schedule.")
    parser.add_argument("--epsilon_min", type=float, default=0.02,
                        help="Minimum epsilon value.")
    parser.add_argument("--learning_rate", type=float, default=0.00025,
                        help="Learning rate.")
    parser.add_argument("--save_name", type=str, default="my_experiment",
                        help="Name for saving models/logs.")

    # Environment parameters
    parser.add_argument("--skip_frames", type=int, default=4,
                        help="Number of frames to skip.")
    parser.add_argument("--mini_render", action="store_true",
                        help="Enable mini rendering.")
    parser.add_argument("--max_steps", type=int, default=90000,
                        help="Maximum steps in an episode.")

    # Double_DQN only
    parser.add_argument("--update_freq", type=int, default=1000,
                        help="Target network update frequency (Double_DQN only).")

    # Death parameters (only used in experimental version)
    parser.add_argument("--death_memory_size", type=int, default=8000,
                        help="Memory size for 'death' experiences.")
    parser.add_argument("--death_batch_size", type=int, default=4,
                        help="Batch size for 'death' experiences.")
    parser.add_argument("--death_steps", type=int, default=25,
                        help="Extra steps for training upon 'death'.")
    parser.add_argument("--death_tries", type=int, default=1,
                        help="Number of attempts in 'death' scenario.")
    parser.add_argument("--death_start_weights",
                        type=lambda s: list(map(float, s.split(','))),
                        default="0.4,0.4,0.2",
                        help="Comma-separated starting weights for death (e.g., '0.4,0.4,0.2').")
    parser.add_argument("--death_end_weights",
                        type=lambda s: list(map(float, s.split(','))),
                        default="0.7,0.2,0.1",
                        help="Comma-separated ending weights for death (e.g., '0.7,0.2,0.1').")
    parser.add_argument("--death_weights_ep_transition", type=int, default=650,
                        help="Episode transition for death weight interpolation.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Build path to the target DQN_main.py
    # For example: Sub-Episode/Super_Mario/base/DQN_main.py
    script_dir = os.path.join(os.path.dirname(__file__), args.game, args.version)
    script_path = os.path.join(script_dir, "DQN_main.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # Build the base command list
    command = [
        sys.executable,  # Uses the current Python interpreter
        script_path,
        "--train_episodes", str(args.train_episodes),
        "--gamma", str(args.gamma),
        "--batch_size", str(args.batch_size),
        "--memory_size", str(args.memory_size),
        "--epsilon", str(args.epsilon),
        "--epsilon_decay", str(args.epsilon_decay),
        "--epsilon_min", str(args.epsilon_min),
        "--learning_rate", str(args.learning_rate),
        "--save_name", args.save_name,
        "--skip_frames", str(args.skip_frames),
        "--max_steps", str(args.max_steps),
        # "--test_episodes", str(args.test_episodes),
        "--training_algorithm", args.training_algorithm  # new argument for algorithm choice
    ]
    
    # Optionally add mini_render flag if specified
    if args.mini_render:
        command.append("--mini_render")
    
    # Only add update_freq if training algorithm is Double_DQN
    if args.training_algorithm == "Double_DQN":
        command.extend(["--update_freq", str(args.update_freq)])
    
    # Only if version is experimental, then add death parameters
    if args.version == "experimental":
        command.extend([
            "--death_memory_size", str(args.death_memory_size),
            "--death_batch_size", str(args.death_batch_size),
            "--death_steps", str(args.death_steps),
            "--death_tries", str(args.death_tries),
            "--death_start_weights", ",".join(map(str, args.death_start_weights)),
            "--death_end_weights", ",".join(map(str, args.death_end_weights)),
            "--death_weights_ep_transition", str(args.death_weights_ep_transition)
        ])
    
    # Print the final command for debugging
    print("Running command:")
    print(" ".join(command))
    
    # Run the selected DQN_main.py script with the constructed arguments
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
