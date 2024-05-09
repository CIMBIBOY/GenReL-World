import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dropout_rate=0.2):
        super(ActorNetwork, self).__init__()
        self.gru = nn.GRU(state_size, hidden_size, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=1)
        )
        self.hidden_state = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)

    def forward(self, state, hidden=None):
        if hidden is None:
            hidden = self.hidden_state.expand(1, state.size(0), -1).contiguous()
        output, hidden = self.gru(state, hidden)
        output = self.batch_norm(output[:, -1, :])
        output = self.dropout(output)
        action = self.fc(output).unsqueeze(1)
        return action, hidden

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, dropout_rate=0.3):
        super(CriticNetwork, self).__init__()
        self.gru = nn.GRU(state_size, hidden_size, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.hidden_state = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)

    def forward(self, state, hidden=None):
        if hidden is None:
            hidden = self.hidden_state.expand(1, state.size(0), -1).contiguous()
        output, hidden = self.gru(state, hidden)
        output = self.batch_norm(output[:, -1, :])
        output = self.dropout(output)
        value = self.fc(output).unsqueeze(1)
        return value, hidden