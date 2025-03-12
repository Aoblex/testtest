import gymnasium as gym
from agents.trainer import RLTrainer
from models.ppo import PPO

def main():
    # Training parameters
    env_id = "CartPole-v1"
    num_episodes = 30  # Changed default to 30
    max_steps = 1000
    
    # Initialize trainer
    trainer = RLTrainer(
        env_id=env_id,
        agent_class=PPO,
        agent_kwargs={
            'lr': 3e-4,
            'gamma': 0.99,
            'epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'num_epochs': 10
        }
    )
    
    # Train agent
    trainer.train(
        num_episodes=num_episodes,
        max_steps=max_steps
    )

if __name__ == "__main__":
    main() 