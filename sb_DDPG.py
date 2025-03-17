from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
import matplotlib.pyplot as plt
import stock_portfolio_env
import gymnasium as gym
import os

AGENT_SAVES_DIR = "saves/agents"
PLOT_SAVES_DIR = "saves/plots"
AGENT_NAME = "DDPG_agent"

def load_agent(
    env: VecEnv,
    agent_saves_dir: str,
    agent_file_name: str,
) -> DDPG:
    """Load the agent if it exists, otherwise create a new one, train it and save it"""
    os.makedirs(agent_saves_dir, exist_ok=True)
    agent_file_path = os.path.join(agent_saves_dir, f"{agent_file_name}.zip")

    if os.path.exists(agent_file_path):
        DDPG_agent = DDPG.load(agent_file_path, device="cpu")
    else:
        DDPG_agent = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-5,
            buffer_size=10000,
            verbose=0,
            tensorboard_log=None,
            device="cpu",
        )
        DDPG_agent.learn(
            total_timesteps=20000,
            progress_bar=True,
        )
        DDPG_agent.save(agent_file_path)
    return DDPG_agent

def plot_results(
    results: dict,
    plot_saves_dir: str,
    plot_file_name: str,
    y_label: str,
) -> None:
    """Plot the results and save it to the plot_file_path"""
    os.makedirs(plot_saves_dir, exist_ok=True)
    plot_file_path = os.path.join(plot_saves_dir, f"{plot_file_name}_{y_label}.png")
    y_values = results[y_label]
    plt.plot(y_values)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.title(f"{y_label} over time using {AGENT_NAME}")
    plt.savefig(plot_file_path)
    plt.close()


def main():
    """Training the agent"""
    # Creating the environment
    env_kwargs={
        "start": "2024-01-01",
        "end": "2024-12-31",
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        "initial_balance": 1000000,
        "transaction_fee_percent": 0.001,
        "reward_scaling": 1e-4,
        "max_shares_norm": 100,
        "render_mode": None,
        "proxy": "http://127.0.0.1:7890",
        "cache_dir": "saves/data",
        "window_size": 10,
    }

    env = make_vec_env(
        "stock_portfolio_env/MultiStockEnvTrain-v0",
        n_envs=4,
        env_kwargs=env_kwargs,
    )

    # Loading the agent
    DDPG_agent = load_agent(
        env,
        AGENT_SAVES_DIR,
        AGENT_NAME,
    )

    """Test the agent"""
    single_env = make_vec_env(
        "stock_portfolio_env/MultiStockEnvTrain-v0",
        n_envs=1,
        env_kwargs=env_kwargs,
    )

    observation = single_env.reset()
    done = False
    while not done:
        action, _states = DDPG_agent.predict(observation, deterministic=True)
        observation, reward, dones, info = single_env.step(action)
        done = dones[0]

    single_env.close()

    """Plot the results"""
    results = single_env.get_attr("results")[0]
    plot_results(
        results,
        PLOT_SAVES_DIR,
        AGENT_NAME,
        "portfolio_values",
    )
    
if __name__ == "__main__":
    main()