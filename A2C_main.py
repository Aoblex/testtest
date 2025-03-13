from agents import A2CAgent


agent = A2CAgent(
    env_id="CartPole-v1",
)

agent.train(plot=True)