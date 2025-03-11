import gymnasium as gym
from agents.trainer import RLTrainer
from models.ddpg import DDPG

def main():
    # Initialize trainer with DDPG
    trainer = RLTrainer(
        env_id="Pendulum-v1",  # Continuous action space environment
        agent_class=DDPG,
        agent_kwargs={
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'tau': 0.005
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