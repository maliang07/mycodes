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


import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=5,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [2048, 512, 256], "big": [512, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        # print('A', A.shape)
        # print('h', h.shape)
        A = torch.transpose(A, 2, 1)  # KxN
        # print('A', A.shape)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.bmm(A, h)
        logits = self.classifiers(M).squeeze(1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits  # , Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=5,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [2048, 512, 256], "big": [512, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        # print('A',A.shape)
        A = torch.transpose(A, 2, 1)  # KxN
        # print('A',A.shape)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.bmm(A, h)
        # print('M',M.shape)
        logits = torch.empty(A.shape[0], self.n_classes).float().to(device)
        # print('logits',logits.shape)
        # print('M[:,c,:]',M[:,0,:].shape)
        a = self.classifiers[0](M[:, 0, :])
        # print('a',a.shape)
        for c in range(self.n_classes):
            logits[:, c] = self.classifiers[c](M[:, c, :]).squeeze(1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits  # , Y_prob, Y_hat, A_raw, results_dict