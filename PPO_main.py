from agents import PPOAgent

agent = PPOAgent(
    env_id="CartPole-v1",
)

agent.train(plot=True)