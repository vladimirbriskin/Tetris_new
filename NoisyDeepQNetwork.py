import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Noisy linear layer with Factorised Gaussian noise"""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training: 
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                             self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class DeepQNetworkNoisy(nn.Module):
    def __init__(self):
        super(DeepQNetworkNoisy, self).__init__()

        # Input layer
        self.fc1 = NoisyLinear(4, 128)  
        self.bn1 = nn.BatchNorm1d(128)

        # Hidden layers
        self.fc2 = NoisyLinear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.2)  

        self.fc3 = NoisyLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.2)  

        # Output layer
        self.fc4 = NoisyLinear(128, 1)  

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x
    
    def reset_noise(self):
        """Reset all noisy layers"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
