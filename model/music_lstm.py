import torch
import torch.nn as nn

from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device


class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(MusicLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.input_size)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.input_size)
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.Wout_root = nn.Linear(self.hidden_size, CHORD_ROOT_SIZE)
        self.Wout_attr = nn.Linear(self.hidden_size, CHORD_ATTR_SIZE)

    def forward(self, x_root, x_attr, feature_key):
        x_root = self.embedding_root(x_root)
        x_attr = self.embedding_attr(x_attr)
        x = x_root + x_attr
        
        feature_key_padded = torch.full((x.shape[0], 1, self.input_size), feature_key.item())
        x = torch.cat([x, feature_key_padded], dim=1)

        lstm_out, _ = self.lstm(x)
        
        if IS_SEPERATED:  
            y_root = self.Wout_root(lstm_out)
            y_attr = self.Wout_attr(lstm_out)
            return y_root, y_attr
        else:
            y = lstm_out
            return y
