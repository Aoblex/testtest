from agents import REINFORCEAgent

agent = REINFORCEAgent(
    env_id="CartPole-v1",
    num_epochs=200,
    num_steps=500,
)

agent.train(
    plot=True
)