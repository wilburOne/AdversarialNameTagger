import torch
import torch.nn as nn


class BatchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional=True, batch_first=True):
        super(BatchLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=bidirectional, batch_first=batch_first
        )

    def forward(self, input, seq_len):
        seq_len, idx = seq_len.sort(descending=True)
        input = input[idx]

        lstm_char_emb = torch.nn.utils.rnn.pack_padded_sequence(
            input, seq_len.data.cpu().numpy(), batch_first=True
        )
        char_lstm_out, char_lstm_h = self.lstm(
            lstm_char_emb
        )
        char_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            char_lstm_out, batch_first=True
        )

        _, unsorted_idx = idx.sort()
        char_lstm_out = char_lstm_out[unsorted_idx]

        return char_lstm_out, char_lstm_h

