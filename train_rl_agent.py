import os
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import stock_portfolio_env

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
        return AlgorithmClass.load(agent_path, env=env, device="cpu")
    
    print(f"Creating new {algorithm} agent")
    return AlgorithmClass(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        device="cpu",
        **params
    )

def train_algorithm(algorithm, args):
    """Train a single algorithm and return its evaluation results"""
    print(f"\n{'='*50}")
    print(f"Processing {algorithm.upper()} algorithm")
    print(f"{'='*50}")
    
    # Setup paths
    agent_file = os.path.join(AGENT_SAVES_DIR, f"{algorithm}_agent.zip")
    
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
        "stock_portfolio_env/MultiStockEnv-v0",
        n_envs=args.n_envs,
        env_kwargs=train_env_kwargs,
    )
    
    # Create or load agent
    agent = create_agent(algorithm, train_env, agent_file if not args.force_train else None)
    
    # Train if needed
    if not os.path.exists(agent_file) or args.force_train:
        print(f"Training {algorithm} on data from {args.train_start_date} to {args.train_end_date}...")
        checkpoint_callback = CheckpointCallback(
            save_freq=args.timesteps // 10,
            save_path=AGENT_SAVES_DIR,
            name_prefix=f"{algorithm}_checkpoint"
        )
        agent.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=checkpoint_callback
        )
        agent.save(agent_file)
        print(f"Agent saved to {agent_file}")
    else:
        print(f"Using existing {algorithm} agent from {agent_file}")
    
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
    print(f"Evaluating {algorithm} on data from {args.eval_start_date} to {args.eval_end_date}...")
    eval_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnv-v0",
        n_envs=1,
        env_kwargs=eval_env_kwargs,
    )
    
    observation = eval_env.reset()
    done = False
    while not done:
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, dones, info = eval_env.step(action)
        done = dones[0]
    
    # Get results
    results = eval_env.get_attr("results")[0]
    
    # Generate individual plot if requested
    if args.individual_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(results["portfolio_values"])
        plt.title(f"Portfolio Value using {algorithm.upper()} (Evaluation Period)")
        plt.xlabel("Day")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        
        plot_path = os.path.join(
            PLOT_SAVES_DIR,
            f"{algorithm}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"Individual plot saved to {plot_path}")
    
    # Print performance metrics
    print(f"\n{algorithm.upper()} Performance Metrics:")
    print(f"Time Period: {args.eval_start_date} to {args.eval_end_date}")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print(f"Total Return: ${results['total_return']:.2f} " +
          f"({(results['total_return']/results['initial_balance'])*100:.2f}%)")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total Transaction Cost: ${results['total_transaction_cost']:.2f}")
    
    # Clean up
    eval_env.close()
    train_env.close()
    
    return {
        'algorithm': algorithm,
        'portfolio_values': results["portfolio_values"],
        'final_value': results["final_portfolio_value"],
        'total_return': results["total_return"],
        'sharpe_ratio': results["sharpe_ratio"],
        'total_trades': results["total_trades"],
        'transaction_cost': results["total_transaction_cost"],
    }

def normalize_portfolio_values(portfolio_values, initial_balance):
    """Normalize portfolio values as percentage of initial balance"""
    return [value / initial_balance * 100 for value in portfolio_values]

def create_comparative_plot(all_results, args):
    """Create a comparative plot with results from multiple algorithms"""
    plt.figure(figsize=(14, 8))
    
    # Plot settings
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    
    # Sort results by performance if needed
    if args.sort_results:
        all_results.sort(key=lambda x: x['final_value'], reverse=True)
    
    # Plot each algorithm's portfolio values
    for i, result in enumerate(all_results):
        color_idx = i % len(colors)
        style_idx = i % len(linestyles)
        
        if args.normalize_values:
            # Plot as percentage of initial balance
            values = normalize_portfolio_values(result['portfolio_values'], args.initial_balance)
            plt.plot(values, color=colors[color_idx], linestyle=linestyles[style_idx], 
                     label=f"{result['algorithm'].upper()} ({values[-1]:.2f}%)")
        else:
            # Plot raw values
            plt.plot(result['portfolio_values'], color=colors[color_idx], linestyle=linestyles[style_idx], 
                     label=f"{result['algorithm'].upper()} (${result['portfolio_values'][-1]:.2f})")
    
    # Add plot details
    title_suffix = "- Normalized (%)" if args.normalize_values else ""
    plt.title(f"Comparative Performance {title_suffix} - Evaluation Period: {args.eval_start_date} to {args.eval_end_date}")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value (%)" if args.normalize_values else "Portfolio Value ($)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    filename = f"comparison_{args.eval_start_date}_{args.eval_end_date}"
    if args.normalize_values:
        filename += "_normalized"
        
    plot_path = os.path.join(PLOT_SAVES_DIR, f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nComparative plot saved to {plot_path}")
    
    # Create performance summary table
    create_performance_table(all_results, args)

def create_performance_table(all_results, args):
    """Create and save a table of performance metrics"""
    plt.figure(figsize=(12, len(all_results) * 0.5 + 2))
    plt.axis('off')
    
    headers = ['Algorithm', 'Final Value ($)', 'Return ($)', 'Return (%)', 'Sharpe', 'Trades', 'Costs ($)']
    
    # Sort results by performance if needed
    if args.sort_results:
        all_results.sort(key=lambda x: x['final_value'], reverse=True)
    
    # Prepare table data
    cell_data = []
    for result in all_results:
        row = [
            result['algorithm'].upper(),
            f"{result['final_value']:,.2f}",
            f"{result['total_return']:,.2f}",
            f"{(result['total_return']/args.initial_balance)*100:.2f}%",
            f"{result['sharpe_ratio']:.4f}",
            str(result['total_trades']),
            f"{result['transaction_cost']:,.2f}"
        ]
        cell_data.append(row)
    
    # Create the table
    table = plt.table(
        cellText=cell_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(headers)
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title(f"Performance Comparison - {args.eval_start_date} to {args.eval_end_date}", fontsize=14, pad=20)
    
    # Save the table
    table_path = os.path.join(PLOT_SAVES_DIR, f"performance_table_{args.eval_start_date}_{args.eval_end_date}.png")
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()
    
    print(f"Performance summary table saved to {table_path}")

def train_and_evaluate(args):
    """Train multiple algorithms and compare their performance"""
    # Ensure directories exist
    os.makedirs(AGENT_SAVES_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVES_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Process selected algorithms
    algorithms = args.algorithms.split(',')
    print(f"Processing {len(algorithms)} algorithms: {', '.join([a.upper() for a in algorithms])}")
    
    # Collect results from all algorithms
    all_results = []
    for algorithm in algorithms:
        if algorithm.lower() in ALGORITHMS:
            result = train_algorithm(algorithm.lower(), args)
            all_results.append(result)
        else:
            print(f"Warning: Algorithm '{algorithm}' not recognized and will be skipped.")
    
    # Create comparative plot if we have results from multiple algorithms
    if len(all_results) > 1:
        create_comparative_plot(all_results, args)
    elif len(all_results) == 1 and not args.individual_plots:
        # Create individual plot if not already created
        plt.figure(figsize=(12, 6))
        plt.plot(all_results[0]['portfolio_values'])
        plt.title(f"Portfolio Value using {all_results[0]['algorithm'].upper()} (Evaluation Period)")
        plt.xlabel("Day")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        
        plot_path = os.path.join(
            PLOT_SAVES_DIR,
            f"{all_results[0]['algorithm']}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
        )
        plt.savefig(plot_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate RL algorithms for trading")
    
    # Algorithm selection
    parser.add_argument('--algorithms', type=str, default='a2c,ppo,ddpg,td3',
                        help='Comma-separated list of algorithms to use (a2c,ppo,ddpg,td3)')
    
    # Plotting options
    parser.add_argument('--individual-plots', action='store_true',
                        help='Generate individual plots for each algorithm')
    parser.add_argument('--normalize-values', action='store_true',
                        help='Normalize portfolio values as percentage of initial balance')
    parser.add_argument('--sort-results', action='store_true',
                        help='Sort algorithms by performance in plots and tables')
    
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