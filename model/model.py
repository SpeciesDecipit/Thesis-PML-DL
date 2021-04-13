import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, embedding_dim, conv_out_dim, lstm_hidden, n_labels, filter_size=3, drop_rate_conv=0.3,
                 drop_rate_lstm=0.5):
        super(PredictionHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim,
                      conv_out_dim,
                      filter_size,
                      padding=filter_size // 2
                      ),
            nn.ReLU(),
            nn.Dropout(drop_rate_conv)
        )
        self.biLSTM = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.out = nn.Linear(2 * lstm_hidden, n_labels)
        self.dropout = nn.Dropout(drop_rate_lstm)

    def forward(self, x):
        # Input: Tensor([batch_size, embedding_dim, max_seq])

        # Expected: Tensor([batch_size, conv_out_dim, max_seq])
        x = self.conv(x)
        x = self.dropout(x)

        # Expected: Tensor([batch_size, max_seq, conv_out_dim])
        x = x.permute(0, 2, 1)

        # Expected: Tensor([batch_size, max_seq, 2*hidden_size])
        outputs, (hn, cn) = self.biLSTM(x)
        cat = torch.cat([hn[0], hn[1]], dim=-1)
        return self.out(self.dropout(cat))
