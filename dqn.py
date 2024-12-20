from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return self.fc2(x)
    
class DQNImage(nn.Module):
    def __init__(self, state_dim, action_discrete_dim_1, action_discrete_dim_2, hidden_dim=256):
        super(DQNImage, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, action_discrete_dim)
        # self.fc3 = nn.Linear(hidden_dim, action_continuous_dim)
        self.fc2 = nn.Linear(hidden_dim, action_discrete_dim_1)
        self.fc3 = nn.Linear(hidden_dim, action_discrete_dim_2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output_discrete_1 = self.fc2(x)  # Saída para números discretos
        output_discrete_2 = self.fc3(x)

        # output_continuous = self.sigmoid(output_continuous)  # Saída para números contínuos
        return output_discrete_1, output_discrete_2 