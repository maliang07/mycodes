import torch.nn as nn

class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)  # seems more common to have Transformer MLP drouput here?

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super(MyMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden = hidden_features

        self.head1 = nn.Linear(in_features, hidden)
        self.drop = nn.Dropout(p=drop)
        self.head2 = nn.Linear(hidden, hidden)
        self.head3 = nn.Linear(hidden, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
        self._init_weights(self.head1)
        self._init_weights(self.head2)
        self._init_weights(self.head3)

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

    def forward(self, xx):
        xx = self.head1(xx)
        xx = self.norm(xx)
        xx = self.relu(xx)
        xx = self.drop(xx)
        xx = self.head2(xx)
        xx = self.norm2(xx)
        xx = self.relu(xx)
        xx = self.drop(xx)
        xx = self.head3(xx)
        return xx