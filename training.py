import torch
import torch.nn as nn
from tqdm import tqdm
#from apex import amp
from sklearn.metrics import f1_score,confusion_matrix
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd

from models.aggregator import *
from models.MLP import *
from models.MyAttentionLstm import *
from dataset import Prost_Dataset
from utils.GradualWarmupScheduler import  *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models.VIT import *
from models.SwinTransformer import *
fe_selector = FESelector()
def train_epoch(train_loader, model, criterion, optimizer, usingfp16=False):
    model.train()
    train_loader1 = tqdm(train_loader)
    loss_ret = 0
    count2 = 0
    for i, (token, mask, label) in enumerate(train_loader1):
        optimizer.zero_grad()
        label = label.to(device)
        with torch.no_grad():
            index = fe_selector(token, label)
            token = token[:, index, :]
        final_score = model(token, mask)
        loss_com = criterion(final_score, label)
        loss_com.backward()
        optimizer.step()
        loss_ret += loss_com.item()
        train_loader1.set_description('loss: {:.4}'.format(loss_ret / count2))

    return loss_ret


def valid_epoch(valid_loader, model, criterion):
    valid_loader = tqdm(valid_loader)
    model.eval()
    ALL_PREDS= []
    TARGETS = []
    for i, (token, mask, label) in enumerate(valid_loader):
        label = label.to(device)
        with torch.no_grad():
            index = fe_selector(token, label)
            token = token[:, index, :]

            final_score = model(token, mask)
            loss_com = criterion(final_score, label)
            ALL_PREDS.append(final_score.sigmoid().sum(1).detach().round())
            TARGETS.append(label.sum(1))

    ALL_PREDS = torch.cat(ALL_PREDS).cpu()
    TARGETS = torch.cat(TARGETS).cpu()
    qwk3 = cohen_kappa_score(ALL_PREDS, TARGETS, weights='quadratic')
    acc = accuracy_score(ALL_PREDS, TARGETS)
    f1 = f1_score(ALL_PREDS, TARGETS, average=None)
    f11 = f1_score(ALL_PREDS, TARGETS, average='macro')
    print('auc_score1', qwk3, 'acc', acc, 'f1', f1, 'f11', f11)

    return qwk3, acc


def train_(model, train_loader, valid_loader):

    init_lr = 3e-4
    warmup_factor = 10
    n_epochs = 100
    warmup_epo = 1

    optimizer = torch.optim.Adam([{"params": model.fe_aggregator.parameters()}, {"params": model.mlp.parameters()}], lr=init_lr / warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo,
                                       after_scheduler=scheduler_cosine)


    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    best_score = -1000
    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer)

        scheduler.step()
        valid_kappa, acc = valid_epoch(valid_loader, model, criterion)
        current_score = valid_kappa
        if current_score > best_score:
            counter = 0
            best_score = current_score
            torch.save(model.state_dict(), os.path.join('./checkpoints',
                                                        str(fold)+str(best_score)+'_'+str(acc)+'max_len_256_256_resize_256_20x_{}.pth'))

        else:
                counter += 1


from typing import List

class Collate:
    def __call__(self, batch: List[dict]) -> dict:
        label = torch.stack([sample[1] for sample in batch])
        token = [sample[0] for sample in batch]
        features = token[0].shape[1]
        token_l = [len(sample[0]) for sample in batch]
        b = len(token_l)
        max_l = max(token_l)
        out = torch.zeros(b, max_l, features)
        mask = torch.zeros(b, max_l, features)

        return out, mask, label


my_collate_fn = Collate()

not_include = []
#not_include = ['2910b44434274b848553a4ec3db11df8','f75c7cec3ddc9fbff27aca59b01c5bf5',
#              '9d917845ea26f2d2a33790c2a755ef8e','da077ff5258dfb7cd6d604d995de7619',
#             '3790f55cad63053e956fb73027179707','3790f55cad63053e956fb73027179707']

if __name__ == '__main__':
    dataset = 'panda'
    folds = [1, 2, 3, 4, 5]
    filedir='./input/panda/h5'

    my_collate_fn = Collate()

    if dataset == "panda":
        df = pd.read_csv('./pandas.csv')
        rr1 = df.groupby(['isup_grade']).count()
        df = df[~df.image_id.isin(not_include)].reset_index(drop=True)

    for fold in folds:
        train_idx = np.where((df['kfold'] != fold))[0]
        valid_idx = np.where((df['kfold'] == fold))[0]
        train_dataset = Prost_Dataset(df=df, indinces=train_idx, filedir=filedir)
        valid_dataset = Prost_Dataset(df=df, indinces=valid_idx, filedir=filedir)

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=32, shuffle=True,
                collate_fn=my_collate_fn,
                num_workers=8, pin_memory=False, sampler=train_sampler, drop_last=False)

        valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=32, shuffle=False,
                collate_fn=my_collate_fn,
                num_workers=8, pin_memory=False, sampler=train_sampler, drop_last=False)

        fe_extractor = nn.Identity()
        fe_aggregator = LSTM_AGG(embeding_l=2048, num_layers =2, hidden_layer=1024, bidirectional=True, batch_first=True)
        #fe_aggregator = vit_base_efficientnet_b3(pretrained=True)
        #fe_aggregator = swin_s3_small_224(pretrained=True)
        mlp = MyMlp(in_features=2048*2, hidden_features=2048, out_features=5)

        model = MyAttentionLstm(fe_extractor=fe_extractor, fe_aggregator=fe_aggregator, mlp=mlp)
        model = model.cuda()

        train_(model, train_loader, valid_loader)
