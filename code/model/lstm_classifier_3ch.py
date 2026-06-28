import torch
import torch.nn as nn
import config.settings as settings

class LSTM_CL(nn.Module):
    def __init__(self, dataloader, user):
        super(LSTM_CL, self).__init__()
        self.name = "lstm_classifier_divided"
        self.user = user
        hp = settings.LSTM_CLASSIFIER
        latent_size = hp["latent_size"]
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader
        self.converged = False
        self.fed_steps = 0
        hidden_size = 16
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=194, hidden_size=hidden_size, num_layers=1, batch_first=True),
            nn.LSTM(input_size=146, hidden_size=hidden_size, num_layers=1, batch_first=True),
            nn.LSTM(input_size=98, hidden_size=hidden_size, num_layers=1, batch_first=True),
        ])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.hid = nn.Sequential(
            nn.Linear(hidden_size * 3, 32), nn.Dropout(p=0.2), nn.ReLU(),
            nn.Linear(32, 32), nn.Dropout(p=0.2), nn.ReLU(),
            nn.Linear(32, latent_size),
        )
        self.classification = nn.Linear(latent_size, settings.CLASS)

    def forward(self, features):
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm_layers):
            if i == 3: break
            last, (_, _) = lstm(features[i])
            lstm_outputs.append(last[:, -1, :])
        x = torch.stack(lstm_outputs, dim=1)
        x, attn_weights = self.attention(x, x, x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.hid(x)
        x = self.classification(x)
        return x, attn_weights
