from agents import A2CAgent


agent = A2CAgent(
    env_id="Ant-v5",
    num_steps=100,
    num_epochs=5000,
)

agent.train(plot=True)