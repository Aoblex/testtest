import gymnasium as gym
from agents.trainer import RLTrainer
from models.reinforce import REINFORCE

def main():
    # Initialize trainer with REINFORCE
    trainer = RLTrainer(
        env_id="CartPole-v1",
        agent_class=REINFORCE,
        agent_kwargs={
            'lr': 1e-3,
            'gamma': 0.99
        },
        seed=42
    )
    
    # Train
    trainer.train(
        num_episodes=2000,  # REINFORCE typically needs more episodes
        eval_interval=20,
        num_eval_episodes=5,
        save_interval=100
    )

if __name__ == "__main__":
    main() 