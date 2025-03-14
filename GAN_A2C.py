from agents import A2CAgent
from agents import Discriminator
import torch
agent = A2CAgent(
    env_id="CartPole-v1",
)
#加载权重
agent.model.load_state_dict(
            torch.load(r"E:\pythonProgram\testtest\results\CartPole-v1\A2C\20250313_144556\A2C_100.pth")
    )
#agent.train(plot=True)
#agent.train_GAN(plot=True)
action_dim=1000
device=torch.device("cpu")

discriminator = Discriminator(input_dim=action_dim).to(device)


true_actions=torch.rand(action_dim)
agent.train_GAN(true_actions=true_actions, discriminator=discriminator)
