"""
	Simple feed forward neural network as an actor and critic networks in PPO agent.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-hidden-hidden-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, hiddens=64):
		"""
			Initialization of the network and setup of the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, hiddens)
		self.layer2 = nn.Linear(hiddens, hiddens)
		self.layer3 = nn.Linear(hiddens, out_dim)

	def forward(self, obs):
		"""
			Forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
	

class LstmNN(nn.Module):
    """
        A standard in_dim-hidden-hidden-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim, hiddens=64, dropout=0.3):
        """
            Initialization of the network and setup of the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(LstmNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, hiddens, batch_first=False)
        self.linear = nn.Linear(hiddens, hiddens)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hiddens, out_dim),
            nn.Tanh(),
            self.dropout
        )

    def forward(self, obs):
        """
            Forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
            obs = obs.unsqueeze(0)

        lst_out, _ = self.lstm(obs)
        output = self.fc(lst_out).squeeze(0)
        
        return output