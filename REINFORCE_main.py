from agents import REINFORCEAgent

agent = REINFORCEAgent(
    env_id="CartPole-v1",
)

agent.train(
    num_epochs=50,
    num_steps=1000,
    plot=True
)