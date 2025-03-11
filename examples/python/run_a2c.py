import gymnasium as gym
from agents.trainer import RLTrainer
from models.a2c import A2C

def main():
    # Initialize trainer with A2C
    trainer = RLTrainer(
        env_id="CartPole-v1",
        agent_class=A2C,
        agent_kwargs={
            'lr': 3e-4,
            'gamma': 0.99,
            'value_coef': 0.5,
            'entropy_coef': 0.01
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