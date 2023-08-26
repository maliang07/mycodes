import torch.nn as nn

class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super(MyMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)  # seems more common to have Transformer MLP drouput here?
        self.norm = nn.BatchNorm1d(hidden_features)
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x