from torch import nn

class SimplePolicy(nn.Module):

    def __init__(self, obs_dim, action_dim):
        self.relu = nn.ReLU6()
        self.nn1 = nn.Linear(obs_dim, 64)
        self.nn2 = nn.Linear(64, 64)
        self.nn3 = nn.Linear(64, action_dim)
    
    def forward(self, obs):
        obs = self.relu(self.nn1(obs))
        obs = self.relu(self.nn2(obs))
        return self.nn3(obs)