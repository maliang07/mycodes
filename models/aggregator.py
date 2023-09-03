import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import random
import numpy as np
from feature_selector import *
EPSILON = np.finfo(float).eps
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRU_AGG(nn.Module):
    def __init__(self, embeding_l, num_layers, hidden_layer, bidirectional, batch_first):
        super(GRU_AGG, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_layer
        self.max_linear = nn.Linear(2048, 2048)

        self.relu = nn.ReLU()
        self.GRU = nn.GRU(input_size=embeding_l, hidden_size=embeding_l, batch_first=True)

        self._init_weights(self.max_linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token, mask):

        token = self.max_linear(token)
        out, h_n = self.GRU(token)

        avg_pool = torch.mean(out * mask, 1)
        max_pool, _ = torch.max(out * mask, 1)

        cat_fe = torch.cat([avg_pool, max_pool], dim=-1)

        return cat_fe

class LSTM_AGG(nn.Module):
    def __init__(self, embeding_l, num_layers, hidden_layer, bidirectional, batch_first):
        super(LSTM_AGG, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_layer
        self.attention1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.attention2 = nn.Linear(1024, 1)
        self.max_linear = nn.Linear(2048, 2048)
        self.lstm = nn.LSTM(embeding_l, hidden_layer, num_layers=num_layers, \
                            bidirectional=bidirectional, batch_first=batch_first)

        self._init_weights(self.attention1)
        self._init_weights(self.attention2)

    def concrete_dropout_neuron(self, dropout_p, temp=1.0 / 10.0, **kwargs):
        unif_noise = Variable(dropout_p.data.new().resize_as_(dropout_p.data).uniform_())
        approx = (
                torch.log(dropout_p + EPSILON)
                - torch.log(1. - dropout_p + EPSILON)
                + torch.log(unif_noise + EPSILON)
                - torch.log(1. - unif_noise + EPSILON)
        )
        approx_output = torch.sigmoid(approx / temp)
        return 1 - approx_output

    def sampled_from_logit_p(self, token_out):
        drop_p = self.attention1(token_out)
        drop_p = self.relu(drop_p)
        drop_p = self.attention2(drop_p)
        dropout_p = torch.sigmoid(drop_p)
        bern_val = self.concrete_dropout_neuron(dropout_p)
        return bern_val

    def sampled_noisy_input(self, token_out):
        bern_val = self.sampled_from_logit_p(token_out)
        bern_val = bern_val.expand_as(token_out)
        noised_input = token_out * bern_val
        return noised_input


    def forward(self,token, mask):

        h_0 = Variable(torch.zeros(self.num_layers * 2, token.size(0), self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(self.num_layers * 2, token.size(0), self.hidden_size).to(device))

        token = self.max_linear(token)

        h_lstm2, (h_out, _) = self.lstm(token, (h_0, c_0))
        prob = random.random()
        if prob < 0.25:  # 0.15
            h_lstm2 = self.sampled_noisy_input(h_lstm2)

        avg_pool = torch.mean(h_lstm2 * mask, 1)
        max_pool, _ = torch.max(h_lstm2 * mask, 1)

        cat_fe = torch.cat([avg_pool, max_pool], dim=-1)

        return cat_fe


class MAX_AGG(nn.Module):
    def __init__(self):
        super(MAX_AGG, self).__init__()

    def forward(self, token):
        max_pool, _ = torch.max(token, 1)
        return max_pool


class MEAN_AGG(nn.Module):
    def __init__(self):
        super(MEAN_AGG, self).__init__()

    def forward(self, token):
        mean_pool = torch.mean(token, 1)
        return mean_pool


class MAX_MEAN_CAT_AGG(nn.Module):
    def __init__(self):
        super(MAX_MEAN_CAT_AGG, self).__init__()

    def forward(self, token):
        mean_pool = torch.mean(token, 1)
        max_pool, _ = torch.max(token, 1)

        cat_fe = torch.cat([mean_pool, max_pool], dim=-1)
        return cat_fe

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, token, mask):
        return self.gem(token, mask, p=self.p, eps=self.eps)

    def gem(self, token, mask, p=3, eps=1e-6):

        mean_pool = (torch.sum(token.clamp(min=eps).pow(p)*mask, 1)/torch.sum(mask,1)).pow(1. / p)
        return mean_pool



class AttentionMIL(nn.Module):
    def __init__(self, embeding_l, L, D, K):
        super(AttentionMIL, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(embeding_l, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )


    def forward(self, x, mask):

        x = self.feature_extractor_part2(x)  # NxL
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.bmm(A, x)  # KxL
        Y_prob = M.squeeze(1)

        return Y_prob


class RNN_AGG(nn.Module):
    def __init__(self, embeding_l, num_layers, hidden_layer, bidirectional, batch_first):
        super(RNN_AGG, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_layer
        self.max_linear = nn.Linear(2048, 2048)

        self.relu = nn.ReLU()
        self.rnn = nn.RNN(input_size=embeding_l, hidden_size=hidden_layer, batch_first=True, num_layers=num_layers,
                          bidirectional=True)

        self._init_weights(self.max_linear)
        # self._init_weights(self.attention2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, token, mask):

        token = self.max_linear(token)
        out, h_n = self.rnn(token)

        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)

        cat_fe = torch.cat([avg_pool, max_pool], dim=-1)

        return cat_fe
