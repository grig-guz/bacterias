from collections import namedtuple
from numpy import concatenate, square
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
        self.landmarks_linear = nn.Linear(5, 32)
        self.agg_agents_linear = nn.Linear(32, 32)
        self.agg_landmarks_linear = nn.Linear(32, 32)
        self.c_agent_linear = nn.Linear(18, 32)
        self.final_linear = nn.Linear(96, action_dim)
        self.lstm = nn.LSTMCell(96, 96)

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

class GCNAttnPolicy(GCNPolicy):

    def __init__(self, obs_dim, action_dim, sigma):
        super().__init__(obs_dim, action_dim, sigma=sigma)
        self.hid_dim = 128
        self.landmarks_linear = nn.Linear(5 + self.hid_dim, self.hid_dim)
        self.c_agent_linear = nn.Linear(14, self.hid_dim)
        self.agents_linear = nn.Linear(13 + self.hid_dim, self.hid_dim)
        self.multihead = nn.MultiheadAttention(self.hid_dim, 4, batch_first=True)
        self.multihead_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.final_linear_mov = nn.Linear(self.hid_dim, 4)
        self.final_linear_inter = nn.Linear(self.hid_dim, action_dim)

        self.lstm = nn.LSTMCell(self.hid_dim * 2, self.hid_dim)
        self.hx = torch.zeros(1, self.hid_dim)
        self.cx = torch.zeros(1, self.hid_dim)

    def mutate(self):
        super().mutate()
        self.hx = torch.zeros(1, self.hid_dim)
        self.cx = torch.zeros(1, self.hid_dim)

    def forward(self, obs):
        agents_obs, landmarks_obs, c_agent_obs = obs
        agents_obs = torch.tensor(agents_obs).float()
        landmarks_obs = torch.tensor(landmarks_obs).float()
        c_agent_obs = torch.tensor(c_agent_obs).float().unsqueeze(0)

        # Individual agent embedding
        c_agent_obs = self.c_agent_linear(c_agent_obs)
        c_agent_obs = self.relu(c_agent_obs)

        # Embeddings for other agents
        if agents_obs.nelement() == 0:
            agents_obs = torch.zeros(1, self.hid_dim)
        else:
            agents_obs = torch.cat([agents_obs, c_agent_obs.repeat(agents_obs.shape[0], 1)], dim=1)
            agents_obs = self.agents_linear(agents_obs)
            agents_obs = self.relu(agents_obs)

        # Embeddings for landmarks
        if landmarks_obs.nelement() == 0:
            agents_obs = torch.zeros(1, self.hid_dim)
        else:
            landmarks_obs = torch.cat([landmarks_obs, c_agent_obs.repeat(landmarks_obs.shape[0], 1)], dim=1)
            landmarks_obs = self.landmarks_linear(landmarks_obs)
            landmarks_obs = self.relu(landmarks_obs)

        # Shared embedding
        obs = torch.cat([agents_obs, landmarks_obs, c_agent_obs]).unsqueeze(0)

        res, _ = self.multihead(obs, obs, obs, need_weights=True)
        res = res + self.multihead_linear(res)
        res = torch.mean(res, dim=1)
        res = torch.cat([res, c_agent_obs], dim=1)
        self.hx, self.cx = self.lstm(res, (self.hx, self.cx))

        act_mov = self.final_linear_mov(self.hx)
        act_inter = self.final_linear_inter(self.hx)

        return int(torch.argmax(act_mov)), int(torch.argmax(act_inter))
