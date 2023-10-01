
import torch.nn as nn


class AttentionLstm(nn.Module):
    def __init__(self, fe_extractor, fe_aggregator, mlp):
        super(AttentionLstm, self).__init__()
        self.fe_extractor = fe_extractor
        self.fe_aggregator = fe_aggregator
        self.mlp = mlp

    def forward(self, token, mask, label=None, return_att=False):
        token = self.fe_extractor(token)
        if return_att:
            x, attention = self.fe_aggregator(token, mask, label, return_att)
        else:
            x = self.fe_aggregator(token, mask, label, return_att)
        x = self.mlp(x)
        if return_att:
            return x, attention
        else:
            return x