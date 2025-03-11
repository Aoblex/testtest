import gymnasium as gym
from agents.trainer import RLTrainer
from models.ppo import PPO

def main():
    # Initialize trainer with PPO
    trainer = RLTrainer(
        env_id="CartPole-v1",
        agent_class=PPO,
        agent_kwargs={
            'lr': 3e-4,
            'gamma': 0.99,
            'epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'num_epochs': 10,
            'batch_size': 64
        },
        seed=42
    )
    
    # Train
    trainer.train(
        num_episodes=1000,
        eval_interval=10,
        num_eval_episodes=5,
        save_interval=100
    )

if __name__ == "__main__":
    main() 