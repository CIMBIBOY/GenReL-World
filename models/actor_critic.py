import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )

    def forward(self, state, hidden):
        output, (h_n, c_n) = self.lstm(state, hidden)
        action = self.fc(output[:, -1, :])
        return action, (h_n, c_n)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(CriticNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, hidden):
        output, (h_n, c_n) = self.lstm(state, hidden)
        value = self.fc(output[:, -1, :])
        return value, (h_n, c_n)