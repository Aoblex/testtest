import os
import sys
import subprocess

if __name__ == "__main__":
    # Get any additional args passed to this script
    args = " ".join(sys.argv[1:])
    # Run the main script with PPO algorithm
    subprocess.run(f"python train_rl_agent.py --algorithm ppo {args}", shell=True) 