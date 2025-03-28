#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test script for RL experiments (loads a trained model and runs test episodes)."
    )
    # Required environment/algorithm arguments:
    parser.add_argument("--game", type=str, required=True,
                        choices=["Car_Racing", "Super_Mario"],
                        help="Which game folder to use.")
    # Default to 'base' so the user doesn't have to specify it, but they can override:
    parser.add_argument("--version", type=str, default="base",
                        choices=["base", "experimental"],
                        help="Which version folder to use (base or experimental). Default is 'base'.")
    parser.add_argument("--training_algorithm", type=str, required=True,
                        choices=["DQN", "Double_DQN", "Dueling_DQN"],
                        help="Which training algorithm to use.")

    # Two primary testing parameters:
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model (.pt file) to load.")
    parser.add_argument("--test_epsilon", type=float, default=0.0,
                        help="Epsilon value for testing (0 = purely greedy).")

    # Optionally let the user control how many episodes to test (default=5):
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of episodes to run in test mode (default 5).")

    return parser.parse_args()

def main():
    args = parse_args()

    # Build the path to the same DQN_main.py script you use for training:
    script_dir = os.path.join(os.path.dirname(__file__), args.game, args.version)
    script_path = os.path.join(script_dir, "DQN_main.py")

    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        sys.exit(1)

    # We'll pass a 'test' flag to let DQN_main know it should do testing
    command = [
        sys.executable,
        script_path,
        "--training_algorithm", str(args.training_algorithm),
        "--mode", "test",  # We assume your DQN_main.py looks for this to switch into test mode
        "--model_path", str(args.model_path),
        "--epsilon", str(args.test_epsilon),
        "--test_episodes", str(args.test_episodes)
    ]

    print("Running test command:")
    print(" ".join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
