import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path

from agents.trainer import RLTrainer
from models.a2c import A2C
from models.ppo import PPO
from models.reinforce import REINFORCE
from models.ddpg import DDPG

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents on gymnasium environments')
    
    # Environment
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Gymnasium environment ID')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Algorithm
    parser.add_argument('--algo', type=str, default='ppo', choices=['a2c', 'ppo', 'reinforce', 'ddpg'],
                       help='RL algorithm to use')
    
    # Training
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--eval-interval', type=int, default=10,
                       help='Episodes between evaluations')
    parser.add_argument('--num-eval-episodes', type=int, default=5,
                       help='Number of episodes for evaluation')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Episodes between saving checkpoints')
    
    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    
    # Algorithm-specific parameters
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient (for A2C/PPO)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy loss coefficient (for A2C/PPO)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                       help='PPO clip parameter')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='DDPG soft update parameter')
    
    # Saving
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--load-path', type=str, default=None,
                       help='Path to load checkpoint from')
    
    return parser.parse_args()

def get_agent_class(algo: str):
    return {
        'a2c': A2C,
        'ppo': PPO,
        'reinforce': REINFORCE,
        'ddpg': DDPG
    }[algo.lower()]

def get_agent_kwargs(args, algo: str):
    base_kwargs = {
        'lr': args.lr,
        'gamma': args.gamma,
        'device': args.device
    }
    
    if algo in ['a2c', 'ppo']:
        base_kwargs.update({
            'value_coef': args.value_coef,
            'entropy_coef': args.entropy_coef
        })
        
    if algo == 'ppo':
        base_kwargs['epsilon'] = args.epsilon
        
    if algo == 'ddpg':
        # Test environment for continuous action space
        env = gym.make(args.env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"DDPG requires continuous action space, but {args.env} has {type(env.action_space)}")
        max_action = float(env.action_space.high[0])
        env.close()
        
        base_kwargs.update({
            'max_action': max_action,
            'lr_actor': args.lr,
            'lr_critic': args.lr * 10,  # Typically higher
            'tau': args.tau
        })
        
    return base_kwargs

def main():
    args = parse_args()
    
    # Get agent class and kwargs
    agent_class = get_agent_class(args.algo)
    agent_kwargs = get_agent_kwargs(args, args.algo)
    
    # Initialize trainer
    trainer = RLTrainer(
        env_id=args.env,
        agent_class=agent_class,
        agent_kwargs=agent_kwargs,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    # Load checkpoint if specified
    if args.load_path:
        trainer.load_checkpoint(args.load_path)
        print(f"Loaded checkpoint from {args.load_path}")
    
    # Train
    print(f"Training {args.algo.upper()} on {args.env}")
    print(f"Agent parameters: {agent_kwargs}")
    
    trainer.train(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_interval=args.save_interval
    )

if __name__ == '__main__':
    main() 