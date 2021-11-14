from pettingzoo.utils.agent_selector import agent_selector
import torch
from torch import nn


class PetriPolicy(nn.Module):

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def mutate(self):
        for _, W in self.state_dict().items():
            eps = torch.randn(size=W.shape)
            W += eps * self.sigma

class SimplePolicy(PetriPolicy):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.relu = nn.ReLU6()
        self.nn1 = nn.Linear(obs_dim, 64)
        self.nn2 = nn.Linear(64, 64)
        self.nn3 = nn.Linear(64, action_dim)
    
    def forward(self, obs):
        obs = self.relu(self.nn1(obs))
        obs = self.relu(self.nn2(obs))
        return self.nn3(obs)


class GCNPolicy(PetriPolicy):

    def __init__(self, obs_dim, action_dim, sigma):
        super().__init__(sigma)
        self.relu = nn.ReLU()
        self.agents_linear = nn.Linear(13, 32)
        self.landmarks_linear = nn.Linear(6, 32)
        self.agg_agents_linear = nn.Linear(32, 32)
        self.agg_landmarks_linear = nn.Linear(32, 32)
        self.c_agent_linear = nn.Linear(13, 32)

        self.final_linear = nn.Linear(96, action_dim)


    def forward(self, obs):
        agents_obs, landmarks_obs, c_agent_obs = obs
        agents_obs = torch.tensor(agents_obs).float()
        landmarks_obs = torch.tensor(landmarks_obs).float()
        c_agent_obs = torch.tensor(c_agent_obs).float()

        if agents_obs.nelement() == 0:
            agents_obs = torch.zeros(32)
        else:
            agents_obs = self.agg_func(agents_obs, self.agents_linear, self.agg_agents_linear)

        if landmarks_obs.nelement() == 0:
            landmarks_obs = torch.zeros(32)
        else:
            landmarks_obs = self.agg_func(landmarks_obs, self.landmarks_linear, self.agg_landmarks_linear)

        c_agent_obs = self.c_agent_linear(c_agent_obs)


        obs = torch.cat([agents_obs, landmarks_obs, c_agent_obs])
        act = self.final_linear(obs)
        return int(torch.argmax(act))

    def agg_func(self, inpt, inner_layer, outer_layer):
        inpt = self.relu(inner_layer(inpt))
        inpt = torch.mean(inpt, dim=0)
        inpt = self.relu(outer_layer(inpt))
        return inpt
