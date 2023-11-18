import os
# os.system('pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python')
#os.system('pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple einops')
import torch
import torch.nn as nn
from tqdm import tqdm
# from apex import amp
from sklearn.metrics import f1_score, confusion_matrix
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from models.aggregator import *
from models.MyMlp import *
from models.attentionLstm import *
from dataset import Prost_Dataset
from utils.GradualWarmupScheduler import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models.VIT import *
from models.SwinTransformer import *
#from models.SwinTransformer import *
#fe_selector = FESelector()

def train_epoch(train_loader, model, criterion, optimizer, feature_selector=None):
    model.train()
    train_loader1 = tqdm(train_loader)
    loss_ret = 0
    count2 = 0
    for i, (token, mask, label) in enumerate(train_loader1):
        print('token',token.shape)
        optimizer.zero_grad()
        label = label.to(device)
        token = token.to(device)
        mask = mask.to(device)
        #print('feature_selector', feature_selector)
        if feature_selector is not None:
            token = feature_selector(token, mask, label)
        final_score = model(token, mask, label)
        loss_com = criterion(final_score, label)
        loss_com.backward()
        optimizer.step()
        loss_ret += loss_com.item()
        #train_loader1.set_description('loss: {:.4}'.format(loss_ret / count2))

    return loss_ret


def valid_epoch(valid_loader, model, criterion, feature_selector=None):
    valid_loader = tqdm(valid_loader)
    model.eval()
    ALL_PREDS= []
    TARGETS = []
    for i, (token, mask, label) in enumerate(valid_loader):
        label = label.to(device)
        token = token.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            if feature_selector is not None:
                token = feature_selector(token, mask, label)
            final_score = model(token, mask, label)
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


def train_(model, train_loader, valid_loader, feature_selector=None):

    init_lr = 3e-4
    warmup_factor = 10
    n_epochs = 100
    warmup_epo = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr / warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo,
                                       after_scheduler=scheduler_cosine)

    criterion = nn.BCEWithLogitsLoss()

    best_score = -1000
    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, feature_selector)

        scheduler.step()
        valid_kappa, acc = valid_epoch(valid_loader, model, criterion, feature_selector)
        current_score = valid_kappa
        if current_score > best_score:
            counter = 0
            best_score = current_score
            torch.save(model.state_dict(), os.path.join('./checkpoints', str(fold)+str(best_score)+'_'+str(acc)+'AttentionMIL_256_20x_{}.pth'))
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
        for i in range(b):
            a = token[i]
            out[i,:a.shape[0],:] = a
            mask[i,:a.shape[0],:] = 1

        return out, mask, label


max_leng = 128
class Collate2:
    def __call__(self, batch: List[dict]) -> dict:
        label = torch.stack([sample[1] for sample in batch])
        token = [sample[0] for sample in batch]
        features = token[0].shape[1]
        token_l = [len(sample[0]) for sample in batch]
        b = len(token_l)
        max_l = max(token_l)
        out = torch.zeros(b, max_leng, features)
        mask = torch.zeros(b, max_leng, 512)

        for i in range(b):
            a = token[i]
            if len(a) >= max_leng:
                out[i, :, :] = a[:max_leng, :]
            else:
                out[i, :a.shape[0], :] = a
            mask[i, :a.shape[0], :] = 1

        return out, mask, label

not_include = ['2910b44434274b848553a4ec3db11df8','f75c7cec3ddc9fbff27aca59b01c5bf5',
              '9d917845ea26f2d2a33790c2a755ef8e','da077ff5258dfb7cd6d604d995de7619',
              '3790f55cad63053e956fb73027179707','3790f55cad63053e956fb73027179707']

if __name__ == '__main__':
    dataset = 'panda'
    folds = [1]
    filedir = './input/h5/panda/'
    agg_type = 'SwinTransformer'#'attention_lstm'#'AttentionMIL'#'CLAM_MB' #''
    embeding_l = 2048
    two_layer = False
    feature_ranking = False
    mlplayer = Mlp2 if two_layer else MyMlp

    if agg_type in ['vit']:
        my_collate_fn = Collate2()
    else:
        my_collate_fn = Collate()

    if dataset == "panda":
        df = pd.read_csv('./input/train_all_final.csv')
        df = df.sample(frac=0.01)
        rr1 = df.groupby(['isup_grade']).count()
        df = df[~df.image_id.isin(not_include)].reset_index(drop=True)

    for fold in folds:
        train_idx = np.where((df['kfold'] != fold))[0]
        valid_idx = np.where((df['kfold'] == fold))[0]
        train_dataset = Prost_Dataset(df=df, indinces=train_idx, filedir=filedir, agg_type=agg_type)
        valid_dataset = Prost_Dataset(df=df, indinces=valid_idx, filedir=filedir, agg_type=agg_type)

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=16, shuffle=True,
                collate_fn=my_collate_fn,
                num_workers=2, pin_memory=False, sampler=train_sampler, drop_last=False)

        valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=16, shuffle=False,
                collate_fn=my_collate_fn,
                num_workers=2, pin_memory=False, sampler=train_sampler, drop_last=False)

        fe_extractor = nn.Identity()
        if agg_type in ['MAX_MEAN_CAT_AGG']:
            fe_aggregator = MAX_MEAN_CAT_AGG()
            mlp = mlplayer(in_features=2048*2, hidden_features=512, out_features=5)
        if agg_type in ['MAX_AGG']:
            fe_aggregator = MAX_AGG()
            mlp = mlplayer(in_features=2048, hidden_features=512, out_features=5)
        if agg_type in ['MEAN_AGG']:
            fe_aggregator = MEAN_AGG()
            mlp = mlplayer(in_features=2048, hidden_features=512, out_features=5)
        if agg_type in ['GeM']:
            fe_aggregator = GeM()
            mlp = mlplayer(in_features=2048, hidden_features=512, out_features=5)
        if agg_type in ['AttentionMIL']:
            fe_aggregator = AttentionMIL(embeding_l=2048, L=500, D=128, K=1)
            mlp = mlplayer(in_features=500, hidden_features=512, out_features=5)
        if agg_type in ['attention_lstm']:
            fe_aggregator = LSTM_AGG(embeding_l=2048, num_layers=2, hidden_layer=1024, bidirectional=True, batch_first=True)
            mlp = mlplayer(in_features=2048 * 2, hidden_features=512, out_features=5)
        if agg_type in ['GRU_AGG']:
            fe_aggregator = GRU_AGG(embeding_l=embeding_l, num_layers=2, hidden_layer=512, bidirectional=True, batch_first=True)
            mlp = mlplayer(in_features=2048 * 2, hidden_features=512, out_features=5)
        if agg_type in ['RNN_AGG']:
            fe_aggregator = RNN_AGG(embeding_l=embeding_l, num_layers=2, hidden_layer=512, bidirectional=True, batch_first=True)
            mlp = mlplayer(in_features=2048 * 2, hidden_features=512, out_features=5)
        if agg_type in ['vit']:
            fe_aggregator = VisionTransformer(embed_dim=512, depth=6, num_heads=8)
            mlp = mlplayer(in_features=512, hidden_features=512, out_features=5)
        if agg_type in ['CLAM_MB']:
            fe_aggregator = CLAM_MB()
            mlp = mlplayer(in_features=512, hidden_features=512, out_features=5)

        if agg_type in ['SwinTransformer']:
            fe_aggregator = swin_s()
            mlp = mlplayer(in_features=512, hidden_features=512, out_features=5)

        model = AttentionLstm(fe_extractor=fe_extractor, fe_aggregator=fe_aggregator, mlp=mlp, agg_type=agg_type)
        if torch.cuda.is_available():
            model = model.cuda()

        feature_selector = None
        if feature_ranking:
            model2 = deepcopy(model)
            model2.load_state_dict(torch.load('./weights/VIT_1.pth'.format(fold), map_location='cpu'), strict=True)
            feature_selector = FESelector(model=model2, n_patch=20, agg_type=agg_type)
            if agg_type in ['vit']:
                model.fe_aggregator.reset_patch(n_patch=20)

        train_(model, train_loader, valid_loader, feature_selector=feature_selector)
