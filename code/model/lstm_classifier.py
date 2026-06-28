import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import config.settings as settings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
import torch.nn.functional as F


# LSTM Auto-Encoder Class
class LSTM_CL(nn.Module):
    def __init__(self, dataloader, user):
        super(LSTM_CL, self).__init__()
        self.name = "lstm_classifier_divided"
        self.user = user
        hp = settings.LSTM_CLASSIFIER
        input_size = hp["input_size"]
        hidden_size = hp["hidden_size"]
        latent_size = hp["latent_size"]
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader
        self.converged = False
        self.fed_steps = 0
        hidden_size = 16
        # LSTM Layer
        self.lstm_layers = nn.ModuleList(
            [
                # for Hourly
                nn.LSTM(
                    input_size=194,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for hourly act
                nn.LSTM(
                    input_size=146,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for hourly loc
                nn.LSTM(
                    input_size=98,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for hourly unlock
                # for Daily
                nn.LSTM(
                    input_size=24,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for daily loc mov
                nn.LSTM(
                    input_size=16,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for daily loc still
                nn.LSTM(
                    input_size=28,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for daily unlock
                nn.LSTM(
                    input_size=10,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                ),  # for daily sleep
            ]
        )
        # self.lstm_layers = nn.ModuleList(
        #     [
        #         #for Hourly
        #         nn.LSTM(input_size=9+2, hidden_size=hidden_size,
        #              num_layers=1, batch_first=True), # for hourly act
        #         #for Daily
        #         nn.LSTM(input_size=31+4, hidden_size=hidden_size,
        #              num_layers=1, batch_first=True), # for daily loc mov

        #     ]
        # )

        # Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=2, batch_first=True
        )

        self.hid = nn.Sequential(
            nn.Linear(hidden_size * 7, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, latent_size),
        )
        # Linear Layer
        self.classification = nn.Linear(latent_size, settings.CLASS)

    def forward(self, features):
        #    LSTM expects (batch_size, sequence_length, input_size)
        lstm_outputs = []
        n = 0
        for i, lstm in enumerate(self.lstm_layers):
            if i == 7:
                break
            last, (_, _) = lstm(features[i])
            lstm_outputs.append(last[:, -1, :])
            # i += 1
        x = torch.stack(lstm_outputs, dim=1)
        x, attn_weights = self.attention(x, x, x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.hid(x)  # (batch_size, linear_output_size)
        x = self.classification(x)  # (batch_size, linear_output_size)
        # return
        return x, attn_weights
