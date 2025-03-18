import os
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import stock_portfolio_env
import numpy as np

# Map algorithm names to their classes
ALGORITHMS = {
    "a2c": A2C,
    "ppo": PPO, 
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
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
    },
    "sac": {
        "learning_rate": 1e-5,
        "buffer_size": 10000,
    },
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

def ensemble_algorithm(args):
    """Create an ensemble of algorithms and evaluate its performance"""
    print(f"\n{'='*50}")
    print(f"Processing ENSEMBLE with method: {args.ensemble_method.upper()}")
    print(f"{'='*50}")
    
    algorithms = args.algorithms.split(',')
    print(f"Creating ensemble using {len(algorithms)} algorithms: {', '.join([a.upper() for a in algorithms])}")
    
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
        "cache_dir": args.data_dir,
        "window_size": args.window_size,
        "time_shift": args.time_shift,
    }
    
    train_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnv-v0",
        n_envs=1,
        env_kwargs=train_env_kwargs,
    )
    
    # Load pretrained models
    agents = {}
    model_weights = {}
    for algorithm in algorithms:
        if algorithm.lower() in ALGORITHMS:
            agent_file = os.path.join(args.agent_dir, f"{algorithm}_agent.zip")
            
            if os.path.exists(agent_file):
                print(f"Loading {algorithm.upper()} model from {agent_file}")
                agents[algorithm] = ALGORITHMS[algorithm].load(agent_file)
                # Default to equal weights
                model_weights[algorithm] = 1.0 / len(algorithms)
            else:
                print(f"Warning: Model file {agent_file} for {algorithm} not found, skipping")
        else:
            print(f"Warning: Algorithm {algorithm} not recognized, skipping")
    
    if not agents:
        raise ValueError("No valid agent models found. Train individual models first.")
    
    # Update weights based on ensemble method
    if args.ensemble_method == 'weighted' or args.ensemble_method == 'rank_weighted':
        print("Evaluating individual models to determine weights...")
        
        # Evaluate each model individually to get their performance metrics
        model_performance = {}
        
        for alg_name, agent in agents.items():
            # Reset environment for each evaluation
            obs = train_env.reset()
            done = False
            
            # Run the model
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, dones, _ = train_env.step(action)
                done = dones[0]
            
            # Get results
            results = train_env.get_attr("results")[0]
            
            # Store performance metrics (using Sharpe ratio or total return)
            if args.ensemble_method == 'weighted':
                model_performance[alg_name] = results["sharpe_ratio"]
                print(f"{alg_name.upper()} Sharpe: {results['sharpe_ratio']:.4f}")
            else:  # rank_weighted
                model_performance[alg_name] = results["total_return"]
                print(f"{alg_name.upper()} Return: ${results['total_return']:,.2f}")
        
        # Update weights based on performance
        if args.ensemble_method == 'weighted':
            # Adjust for negative Sharpe ratios
            min_sharpe = min(model_performance.values())
            if min_sharpe < 0:
                adjusted_performance = {k: v - min_sharpe + 0.1 for k, v in model_performance.items()}
            else:
                adjusted_performance = model_performance
                
            # Calculate weights proportional to performance
            total = sum(adjusted_performance.values())
            if total > 0:  # Avoid division by zero
                model_weights = {k: v / total for k, v in adjusted_performance.items()}
            
        elif args.ensemble_method == 'rank_weighted':
            # Rank models by performance
            ranked_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
            
            # Weights based on rank (1st place gets highest weight)
            rank_weights = {}
            total_rank_value = sum(1 / (i + 1) for i in range(len(ranked_models)))
            
            for i, (alg_name, _) in enumerate(ranked_models):
                rank_weights[alg_name] = (1 / (i + 1)) / total_rank_value
            
            model_weights = rank_weights
    
    elif args.ensemble_method == 'best_only':
        # Find the best performing model
        print("Evaluating individual models to find the best one...")
        
        best_model = None
        best_sharpe = float('-inf')
        
        for alg_name, agent in agents.items():
            # Reset environment
            obs = train_env.reset()
            done = False
            
            # Run the model
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, dones, _ = train_env.step(action)
                done = dones[0]
            
            # Get results
            results = train_env.get_attr("results")[0]
            sharpe = results["sharpe_ratio"]
            
            print(f"{alg_name.upper()} Sharpe: {sharpe:.4f}")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model = alg_name
        
        # Set weights to use only best model
        model_weights = {alg: 1.0 if alg == best_model else 0.0 for alg in agents.keys()}
        print(f"Best model: {best_model.upper()} (Sharpe: {best_sharpe:.4f})")
    
    train_env.close()

    # Print final model weights
    print("\nEnsemble Weights:")
    for alg, weight in model_weights.items():
        print(f"{alg.upper()}: {weight:.4f}")
    
    # Function to combine predictions from multiple models
    def ensemble_predict(observation, deterministic=True):
        actions = {}
        
        # Get actions from all models
        for alg_name, agent in agents.items():
            action, _ = agent.predict(observation, deterministic=deterministic)
            actions[alg_name] = action
        
        # Combine actions based on the chosen method
        if args.ensemble_method == 'voting':
            # For discrete action spaces, use sign voting
            combined_action = np.zeros_like(next(iter(actions.values())))
            
            for alg_name, action in actions.items():
                combined_action += np.sign(action) * model_weights[alg_name]
            
            # Convert to final action by taking the sign
            combined_action = np.sign(combined_action)
        else:
            # Weighted average for continuous actions
            combined_action = np.zeros_like(next(iter(actions.values())))
            
            for alg_name, action in actions.items():
                combined_action += action * model_weights[alg_name]
        
        return combined_action
    
    # Evaluate the ensemble
    print(f"\nEvaluating ensemble on data from {args.eval_start_date} to {args.eval_end_date}...")
    

    eval_env_kwargs = {
        "start": args.eval_start_date,
        "end": args.eval_end_date,
        "tickers": args.tickers.split(','),
        "initial_balance": args.initial_balance,
        "transaction_fee_percent": args.fee,
        "reward_scaling": args.reward_scaling,
        "max_shares_norm": args.max_shares,
        "proxy": args.proxy,
        "cache_dir": args.data_dir,
        "window_size": args.window_size,
        "time_shift": args.time_shift,
    }

    eval_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnv-v0",
        n_envs=1,
        env_kwargs=eval_env_kwargs,
    )
    # Reset evaluation environment
    obs = eval_env.reset()
    done = False
    
    # Run the ensemble model
    while not done:
        action = ensemble_predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        done = dones[0]
    
    # Get results
    results = eval_env.get_attr("results")[0]
    
    # Close environments
    eval_env.close()
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    if args.time_shift != 0:
        time_shift_label = f" (Shift={args.time_shift:+d})"  # +1 or -1 format
    else:
        time_shift_label = ""
    plt.plot(results["portfolio_values"])
    plt.title(f"Portfolio Value using ENSEMBLE ({args.ensemble_method.upper()}){time_shift_label} (Evaluation Period)")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    
    plot_path = os.path.join(
        args.plot_dir,
        f"ensemble_{args.ensemble_method}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    
    # Prepare result data
    final_value = results["final_portfolio_value"]
    total_return = results["total_return"]
    initial_balance = args.initial_balance
    percent_return = (total_return / initial_balance) * 100
    sharpe_ratio = results["sharpe_ratio"]
    total_trades = results["total_trades"]
    transaction_cost = results["total_transaction_cost"]
    
    # Print performance metrics
    print(f"\nPerformance Metrics for ENSEMBLE ({args.ensemble_method.upper()}):")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: ${total_return:,.2f} ({percent_return:.2f}%)")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Transaction Cost: ${transaction_cost:,.2f}")
    
    # Return results for comparative analysis
    return {
        "algorithm": f"ensemble_{args.ensemble_method}",
        "final_value": final_value,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "transaction_cost": transaction_cost,
        "portfolio_values": results["portfolio_values"]
    }

def train_algorithm(algorithm, args):
    """Train a single algorithm and return its evaluation results"""
    print(f"\n{'='*50}")
    print(f"Processing {algorithm.upper()} algorithm")
    print(f"{'='*50}")
    
    # Setup paths
    agent_file = os.path.join(args.agent_dir, f"{algorithm}_agent.zip")
    
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
        "cache_dir": args.data_dir,
        "window_size": args.window_size,
        "time_shift": args.time_shift,
    }
    
    # Create vectorized environment for training
    train_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnv-v0",
        n_envs=args.n_envs,
        env_kwargs=train_env_kwargs,
    )
    
    # Create or load agent
    agent = create_agent(
        algorithm, train_env,
        agent_file if not args.force_train else None
    )
    
    # Train if needed
    if not os.path.exists(agent_file) or args.force_train:
        print(f"Training {algorithm} on data from {args.train_start_date} to {args.train_end_date}...")
        checkpoint_callback = CheckpointCallback(
            save_freq=args.timesteps // 10,
            save_path=args.agent_dir,
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
        "cache_dir": args.data_dir,
        "window_size": args.window_size,
        "time_shift": args.time_shift,
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
        if args.time_shift != 0:
            time_shift_label = f" (Shift={args.time_shift:+d})"  # +1 or -1 format
        else:
            time_shift_label = ""
        plt.plot(results["portfolio_values"])
        plt.title(f"Portfolio Value using {algorithm.upper()}{time_shift_label} (Evaluation Period)")
        plt.xlabel("Day")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        
        plot_path = os.path.join(
            args.plot_dir,
            f"{algorithm}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
        )
        plt.savefig(plot_path)
        plt.close()
    
    # Prepare result data
    final_value = results["final_portfolio_value"]
    total_return = results["total_return"]
    initial_balance = args.initial_balance
    percent_return = (total_return / initial_balance) * 100
    sharpe_ratio = results["sharpe_ratio"]
    total_trades = results["total_trades"]
    transaction_cost = results["total_transaction_cost"]
    
    # Print performance metrics
    print(f"\nPerformance Metrics for {algorithm.upper()}:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: ${total_return:,.2f} ({percent_return:.2f}%)")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Transaction Cost: ${transaction_cost:,.2f}")
    
    # Return results for comparative analysis
    return {
        "algorithm": algorithm,
        "final_value": final_value,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "transaction_cost": transaction_cost,
        "portfolio_values": results["portfolio_values"]
    }

def normalize_portfolio_values(portfolio_values, initial_balance):
    """Normalize portfolio values as percentage of initial balance"""
    return [value / initial_balance * 100 for value in portfolio_values]

def create_comparative_plot(all_results, args):
    """Create and save a comparative plot for multiple algorithms"""
    plt.figure(figsize=(14, 7))
    
    # For returning values as percentage of initial balance
    initial_balance = args.initial_balance
    
    for result in all_results:
        portfolio_values = result["portfolio_values"]
        if args.normalize_values:
            # Convert to percentage of initial balance
            portfolio_values = [v / initial_balance * 100 for v in portfolio_values]
            ylabel = "Portfolio Value (% of Initial)"
        else:
            ylabel = "Portfolio Value ($)"
        
        plt.plot(portfolio_values, label=f"{result['algorithm'].upper()} (Final: {'%.2f%%' % (result['total_return']/initial_balance*100) if args.normalize_values else '${:,.2f}'.format(result['final_value'])})")
    
    plt.title(f"Portfolio Value Comparison (Evaluation Period: {args.eval_start_date} to {args.eval_end_date})")
    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc='upper left')
    
    filename = f"comparison_{args.eval_start_date}_{args.eval_end_date}"
    if args.normalize_values:
        filename += "_normalized"
    plot_path = os.path.join(args.plot_dir, f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nComparative plot saved to {plot_path}")
    
    # Create performance summary table
    create_performance_table(all_results, args)

def create_performance_table(all_results, args):
    """Create and save a table of performance metrics"""
    plt.figure(figsize=(12, len(all_results) * 0.5 + 2))
    plt.axis('off')
    
    headers = ['Algorithm', 'Shift', 'Final Value ($)', 'Return ($)', 'Return (%)', 'Sharpe', 'Trades', 'Costs ($)']
    
    # Sort results by performance if needed
    if args.sort_results:
        all_results.sort(key=lambda x: x['final_value'], reverse=True)
    
    # Prepare table data
    cell_data = []
    for result in all_results:
        row = [
            result['algorithm'].upper(),
            f"{args.time_shift:+d}",  # Show +1 or -1 format
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
    table_path = os.path.join(args.plot_dir, f"performance_table_{args.eval_start_date}_{args.eval_end_date}.png")
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()
    
    print(f"Performance summary table saved to {table_path}")

def train_and_evaluate(args):
    """Train multiple algorithms and compare their performance"""
    # Ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.agent_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Process selected algorithms
    algorithms = args.algorithms.split(',')
    print(f"Processing {len(algorithms)} algorithms: {', '.join([a.upper() for a in algorithms])}")
    
    # Collect results from all algorithms
    all_results = []
    
    # Check if using ensemble
    use_ensemble = args.ensemble_method not in ["", "none", "None"]
    
    # Train/evaluate individual algorithms
    for algorithm in algorithms:
        if algorithm.lower() in ALGORITHMS:
            result = train_algorithm(algorithm.lower(), args)
            all_results.append(result)
        else:
            print(f"Warning: Algorithm '{algorithm}' not recognized and will be skipped.")
    
    # Run ensemble if specified
    if use_ensemble and len(algorithms) > 1:
        print("\nCreating ensemble of trained models...")
        result = ensemble_algorithm(args)
        all_results.append(result)
    
    # Create comparative plot if we have results from multiple sources
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
            args.plot_dir,
            f"{all_results[0]['algorithm']}_portfolio_eval_{args.eval_start_date}_{args.eval_end_date}.png"
        )
        plt.savefig(plot_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate RL algorithms for trading")
    
    # Directory paths
    parser.add_argument('--save-dir', type=str, default="saves",
                        help='Base directory for all saved files')
    parser.add_argument('--agent-dir', type=str, default=None,
                        help='Directory for agent models (default: {save_dir}/agents)')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory for saved plots (default: {save_dir}/plots)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory for cached data (default: {save_dir}/data)')
    
    # Algorithm selection
    parser.add_argument('--algorithms', type=str, default='a2c,ppo,ddpg,td3,sac',
                        help='Comma-separated list of algorithms to use (a2c,ppo,ddpg,td3,sac)')
    
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
    parser.add_argument('--proxy', type=str, default="http://127.0.0.1:7890",
                        help='Proxy server URL (optional)')
    
    # Hyperparameters to tune
    parser.add_argument('--window-size', type=int, default=16,
                        help='Number of days to include in state observation window')
    parser.add_argument('--time-shift', type=int, default=0,
                        help='Temporal shift in days (positive=future data, negative=past data)')

    # Ensemble method
    parser.add_argument('--ensemble-method', type=str, default='average',
                        help='Ensemble method to use (average, weighted, rank_weighted, voting, best_only)')
    
    args = parser.parse_args()
    
    # Set default directories if not specified
    if args.agent_dir is None:
        args.agent_dir = os.path.join(args.save_dir, "agents")
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.save_dir, "plots")
    if args.data_dir is None:
        args.data_dir = os.path.join(args.save_dir, "data")
    
    return args

def main():
    args = parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main() 