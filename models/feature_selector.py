import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import torch


class FESelector(nn.Module):

    def __init__(self, model, n_patch, agg_type='attention_lstm'):
        super(FESelector, self).__init__()
        self.model = model
        self.n_patch = n_patch
        self.agg_type = agg_type

    def forward(self, token, mask, label):
        selected_token = torch.zeros(token.shape[0], self.n_patch, token.shape[-1])#.cuda()
        if torch.cuda.is_available():
            selected_token = selected_token.cuda()

        with torch.no_grad():
            final_score, attention = self.model(token, mask, label, return_att=True)
            attention = attention.squeeze(1)
            # print('attention', attention.shape)
            if self.agg_type in ['attention_lstm']:
                a, indinces = torch.topk(-attention.squeeze(-1), k=self.n_patch, dim=1)
            elif self.agg_type in ['CLAM_MB']:
                attention = attention.transpose(-1, -2)
                attention = attention.mean(-1)
                # print('attention', attention.shape)
                a, indinces = torch.topk(attention.squeeze(-1), k=self.n_patch, dim=1)
            elif self.agg_type in ['vit']:
                # print('attention', attention.shape)
                attention = attention.mean(1).mean(1)[:, 0:attention.shape[-1]-1]
                print('attention', attention.shape)
                a, indinces = torch.topk(attention.squeeze(-1), k=self.n_patch, dim=1)
            else:
                a, indinces = torch.topk(attention.squeeze(-1), k=self.n_patch, dim=1)

            #print('token', token.shape)
            #print('indinces', indinces.shape)
            for i in range(indinces.shape[0]):
                selected_token[i, :, :] = token[i, indinces[i, :], :]
            print('selected_token',selected_token.shape)

        return selected_token
