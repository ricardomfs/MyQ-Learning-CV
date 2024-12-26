from torch import nn
import torch.nn.functional as F

class DQCNN(nn.Module):
    def __init__(self, input_channels, action_dim, hidden_dim=512):
        super(DQCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=64 * 4 * 6, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # "Achata as dimensões para a entrada no layer denso fc"
        x = F.relu(self.fc1(x))

        return self.fc2(x)