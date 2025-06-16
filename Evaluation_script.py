import gymnasium as gym 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.distributions import Categorical
from environment_handler import Environment


env=Environment()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class agent(nn.Module):
    def __init__(self):
        super(agent,self).__init__()
        self.actor=nn.Sequential(
            nn.Conv2d(12,20,5,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(20,40,4,2,1),
            nn.ReLU(),
            nn.Conv2d(40,64,3,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216,512),
            nn.ReLU(),
            nn.Linear(512,200),
            nn.ReLU(),
            nn.Linear(200,5)
        )
        self.critic=nn.Sequential(
            nn.Conv2d(12,20,5,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(20,40,4,2,1),
            nn.ReLU(),
            nn.Conv2d(40,64,3,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    
    def values(self,obs):
        return self.critic(obs)
    
    def get_actions_probs(self,state,action=None):
        dist=Categorical(logits=self.actor(state))
        value=self.critic(state)
        if action is None:
            action=dist.sample()
        log_prob=dist.log_prob(action)
        entropy=dist.entropy()
        return action, log_prob,entropy,value 
    
model=agent().to(device)
checkpoint = torch.load("ppo_model100.pt", map_location=device)
model.actor.load_state_dict(checkpoint['actor_state_dict'])
model.critic.load_state_dict(checkpoint['critic_state_dict'])
model.eval()  # Set to evaluation mode if not training
    
while True:
    state = env.reset().to(device)
    rewards=[]
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = model.get_actions_probs(state.unsqueeze(0))
        state, reward, terminated, truncated= env.input(action.item())
        state=state.to(device)
        rewards.append(reward)
        done = terminated or truncated
    print(f'Reward of an episode {sum(rewards)}')

