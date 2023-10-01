import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
import torch


class FESelector(nn.Module):

    def __init__(self, model, n_patch, agg_type):
        super(FESelector, self).__init__()
        self.model = model
        self.n_patch = n_patch
        self.agg_type = agg_type

    def forward(self, token, mask, label):
        selected_token = torch.zeros(token.shape[0], self.n_patch, token.shape[-1]).cuda()
        with torch.no_grad():
            final_score, attention = self.model(token, mask, label, return_att=True)
            attention = attention.squeeze(1)
            if self.agg_type in ['attention_lstm']:
                a, indinces = torch.topk(-attention.squeeze(-1), k=self.n_patch, dim=1)
            else:
                a, indinces = torch.topk(attention.squeeze(-1), k=self.n_patch, dim=1)

            for i in range(indinces.shape[0]):
                selected_token[i, :, :] = token[i, indinces[i, :], :]

        return selected_token
