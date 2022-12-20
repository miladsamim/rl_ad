import torch 
import torch.nn as nn
import torch.nn.functional as F

class PDriveDQN(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        in_dim = args.h_size*2
        n_acts = args.act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, n_acts)
        )

    def forward(self, states):
        return self.net(states) # -> Q vals for actions