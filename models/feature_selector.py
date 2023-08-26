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

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super(FESelector, self).__init__()

    def mim_rank(self, trainset, label):
        from sklearn.feature_selection import mutual_info_classif
        return mutual_info_classif(trainset, label)

    def enet_rank(self, trainset, valset):
        rank, metrics = self._lasso_enet_common(trainset, valset,
                                                alpha=0.5, name='enet')
        return rank

    def lasso_rank(self, trainset, valset):
        rank, metrics = self._lasso_enet_common(trainset, valset,
                                                alpha=1., name='lasso')
        return rank

    def _lasso_enet_common(self, trainset, testset, alpha, name):
        x_train, y_train = trainset
        new_x_train = x_train.copy().astype(np.float64)
        new_y_train = y_train.copy().astype(np.float64)
        fit = glmnet(x=new_x_train.copy(), y=new_y_train.copy(), family='binomial',
                     alpha=alpha, nlambda=1000)

        def _get_rank_by_soln_path(soln_path_coefs):
            rank = np.zeros(soln_path_coefs.shape[0])

            for f in range(soln_path_coefs.shape[0]):
                for i in range(soln_path_coefs.shape[1]):
                    if soln_path_coefs[f, i] != 0.:
                        rank[f] = -i
                        break

            rank[rank == 0] = -(soln_path_coefs.shape[1])
            return rank

        rank = _get_rank_by_soln_path(fit['beta'])

        return rank

    def rf_rank(self, trainset, label):
        out2 = trainset.reshape(trainset.shape[0], trainset.shape[1] * trainset.shape[2])
        all_train_y_sum = torch.stack(label).sum(1)

        clf = RandomForestClassifier(n_estimators=200, n_jobs=4)
        clf.fit(out2, all_train_y_sum)
        aa = clf.feature_importances_.reshape(trainset.shape[0], 2048)
        bb = np.mean(aa, axis=-1)
        index = np.argsort(bb)

        return index