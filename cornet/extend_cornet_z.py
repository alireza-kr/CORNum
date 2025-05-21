import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import functional as FM
from cornet import cornet_z
from cornet.cornet_z import Flatten, Identity, CORblock_Z
from collections import OrderedDict

class ExtendedCORnet(pl.LightningModule):
    def __init__(self, out_features, pretrained = True, loss = nn.CrossEntropyLoss(), optimizer = 'adam',  lr = 1e-3, map_location=None):
        super(ExtendedCORnet, self).__init__()
        self.model = cornet_z(pretrained = pretrained, map_location = map_location)
        if out_features > 0:
            self.model.module.decoder.linear = nn.Linear(512, out_features)
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer
        self.out_features = out_features

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y, task="multiclass", num_classes=self.out_features)
        return loss, acc

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif self.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
            else:
                raise NameError(f'Unknown optimizer {self.optimizer}')
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

class ZeroMask(nn.Module):
    def __init__(self, mask):
        super(ZeroMask, self).__init__()
        self.mask = mask

    def forward(self, x):
        x[:,self.mask] = 0
        return x

class ShuffleMask(nn.Module):
    def __init__(self, mask):
        super(ShuffleMask, self).__init__()
        self.mask = mask

    def forward(self, x):
        y = x[torch.randperm(x.shape[0])]
        x[:,self.mask] = y[:,self.mask]
        return x
