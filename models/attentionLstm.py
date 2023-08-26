
import torch.nn as nn

class AttentionLstm(nn.Module):
    def __init__(self, fe_extractor, fe_aggregator, mlp):
        super(AttentionLstm, self).__init__()
        self.fe_extractor = fe_extractor
        self.fe_aggregator = fe_aggregator
        self.mlp = mlp

    def forward(self, token, mask):
        token = self.fe_extractor(token)
        x = self.fe_aggregator(token, mask)
        x = self.mlp(x)
        return x