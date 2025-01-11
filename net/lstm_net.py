import torch
import torch.nn as nn
import torch.functional as F

#memo 音素とmspがあれば，f0出るはず．LSTMかGRUか1DConv
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two LSTMs together to form a `stacked LSTM`,
#             with the second LSTM taking in outputs of the first LSTM and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             LSTM layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
#         proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0


class LSTM_net(nn.Module):
    def __init__(self, input_size = 2*80, hidden_size = 128, num_layers = 2, fc_size = 4096, output_size = 80, dropout = 0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_size = fc_size
        self.output_size = output_size
        self.dropout = dropout

        super(LSTM_net, self).__init__()
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first = True, dropout=self.dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_size)
        self.fc2 = nn.Linear(fc_size, self.output_size)
        self.mish = nn.Mish()
        
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        h, _= self.lstm(x)
        h = self.fc1(h.reshape(h.shape[0]*h.shape[1],h.shape[2]))
        h = self.mish(h)
        h = self.fc2(h)
        return h.view(x.shape[0], x.shape[1], self.output_size)


class GRU_net(nn.Module):
    def __init__(self, input_size = 2*80, hidden_size = 128, num_layers = 2, fc_size = 4096, output_size = 80, dropout = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_size = fc_size
        self.output_size = output_size
        self.dropout = dropout

        super(GRU_net, self).__init__()
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first = True, dropout = self.dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_size)
        self.fc2 = nn.Linear(fc_size, self.output_size)
        self.mish = nn.Mish()

    def forward(self, x):
        h, _= self.gru(x)
        h = self.fc1(h.view(h.shape[0]*h.shape[1],h.shape[2]))
        h = self.mish(h)
        h = self.fc2(h)
        return h.view(x.shape[0],x.shape[1],self.output_size)
