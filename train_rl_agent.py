import os
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import stock_portfolio_env
import gymnasium as gym

# Base directories
SAVES_DIR = "saves"
AGENT_SAVES_DIR = os.path.join(SAVES_DIR, "agents")
PLOT_SAVES_DIR = os.path.join(SAVES_DIR, "plots")
DATA_CACHE_DIR = os.path.join(SAVES_DIR, "data")

# Map algorithm names to their classes
ALGORITHMS = {
    "a2c": A2C,
    "ppo": PPO, 
    "ddpg": DDPG,
    "td3": TD3
}

# Default hyperparameters for each algorithm
ALGORITHM_PARAMS = {
    "a2c": {
        "learning_rate": 1e-5,
        "n_steps": 5,
    },
    "ppo": {
        "learning_rate": 1e-5,
        "n_steps": 128,
        "batch_size": 16,
        "n_epochs": 4,
    },
    "ddpg": {
        "learning_rate": 1e-5,
        "buffer_size": 10000,
    },
    "td3": {
        "learning_rate": 1e-5,
        "buffer_size": 10000,
    }
}

def create_agent(algorithm, env, agent_path=None):
    """Create or load an agent"""
    AlgorithmClass = ALGORITHMS[algorithm.lower()]
    params = ALGORITHM_PARAMS[algorithm.lower()]
    
    if agent_path and os.path.exists(agent_path):
        print(f"Loading pre-trained {algorithm} agent from {agent_path}")
        return AlgorithmClass.load(agent_path, env=env)
    
    print(f"Creating new {algorithm} agent")
    return AlgorithmClass(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        device="cpu",
        **params
    )

def train_and_evaluate(args):
    """Train an agent and evaluate its performance"""
    # Ensure directories exist
    os.makedirs(AGENT_SAVES_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVES_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    agent_file = os.path.join(AGENT_SAVES_DIR, f"{args.algorithm}_agent.zip")
    
    # Training environment parameters
    train_env_kwargs = {
        "start": args.train_start_date,
        "end": args.train_end_date,
        "tickers": args.tickers.split(','),
        "initial_balance": args.initial_balance,
        "transaction_fee_percent": args.fee,
        "reward_scaling": args.reward_scaling,
        "max_shares_norm": args.max_shares,
        "proxy": args.proxy,
        "cache_dir": DATA_CACHE_DIR,
        "window_size": args.window_size,
    }
    
    # Create vectorized environment for training
    train_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnvTrain-v0",
        n_envs=args.n_envs,
        env_kwargs=train_env_kwargs,
    )
    
    # Create or load agent
    agent = create_agent(args.algorithm, train_env, agent_file if not args.force_train else None)
    
    # Train if needed
    if not os.path.exists(agent_file) or args.force_train:
        print(f"Training {args.algorithm} on data from {args.train_start_date} to {args.train_end_date}...")
        checkpoint_callback = CheckpointCallback(
            save_freq=args.timesteps // 10,
            save_path=AGENT_SAVES_DIR,
            name_prefix=f"{args.algorithm}_checkpoint"
        )
        agent.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=checkpoint_callback
        )
        agent.save(agent_file)
        print(f"Agent saved to {agent_file}")
    
    # Evaluation environment parameters
    eval_env_kwargs = {
        "start": args.eval_start_date,
        "end": args.eval_end_date,
        "tickers": args.tickers.split(','),
        "initial_balance": args.initial_balance,
        "transaction_fee_percent": args.fee,
        "reward_scaling": args.reward_scaling,
        "max_shares_norm": args.max_shares,
        "proxy": args.proxy,
        "cache_dir": DATA_CACHE_DIR,
        "window_size": args.window_size,
    }
    
    # Create environment for evaluation
    print(f"Evaluating on data from {args.eval_start_date} to {args.eval_end_date}...")
    eval_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnvTrain-v0",
        n_envs=1,
        env_kwargs=eval_env_kwargs,
    )
    
    observation = eval_env.reset()
    done = False
    while not done:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, dones, info = eval_env.step(action)
        done = dones[0]
    
    # Plot results
    results = eval_env.get_attr("results")[0]
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    plt.plot(results["portfolio_values"])
    plt.title(f"Portfolio Value using {args.algorithm.upper()} (Evaluation Period)")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    
    plot_path = os.path.join(
        PLOT_SAVES_DIR,
        f"{args.algorithm}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    
    # Print performance metrics
    print("\nEvaluation Performance Metrics:")
    print(f"Time Period: {args.eval_start_date} to {args.eval_end_date}")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print(f"Total Return: ${results['total_return']:.2f} " +
          f"({(results['total_return']/results['initial_balance'])*100:.2f}%)")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total Transaction Cost: ${results['total_transaction_cost']:.2f}")
    
    print(f"\nResults plot saved to {plot_path}")
    
    # Clean up
    eval_env.close()
    train_env.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate RL algorithms for trading")
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='a2c', choices=ALGORITHMS.keys(),
                        help='RL algorithm to use')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=20000, 
                        help='Number of timesteps to train for')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of environments for parallel training')
    parser.add_argument('--force-train', action='store_true',
                        help='Force training even if a saved model exists')
    
    # Training environment parameters
    parser.add_argument('--train-start-date', type=str, default='2000-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--train-end-date', type=str, default='2024-02-29',
                        help='End date for training data (YYYY-MM-DD)')
    
    # Evaluation environment parameters  
    parser.add_argument('--eval-start-date', type=str, default='2024-03-01',
                        help='Start date for evaluation data (YYYY-MM-DD)')
    parser.add_argument('--eval-end-date', type=str, default='2025-03-01',
                        help='End date for evaluation data (YYYY-MM-DD)')
    
    # Common environment parameters
    parser.add_argument('--tickers', type=str, default='AAPL,MSFT,AMZN,GOOGL,META',
                        help='Comma-separated list of ticker symbols')
    parser.add_argument('--initial-balance', type=float, default=1000000,
                        help='Initial portfolio balance')
    parser.add_argument('--fee', type=float, default=0.001,
                        help='Transaction fee percentage')
    parser.add_argument('--reward-scaling', type=float, default=1e-4,
                        help='Scaling factor for rewards')
    parser.add_argument('--max-shares', type=int, default=100,
                        help='Maximum shares per trade normalization factor')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Number of days to include in state observation window')
    parser.add_argument('--proxy', type=str, default=None,
                        help='Proxy server URL (optional)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main() 