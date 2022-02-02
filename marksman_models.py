import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


class MultiLayerModel1(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, inSize, hiddenSizes, outSize, dtype = None, loss = nn.MSELoss(), optimizer = torch.optim.SGD, bias = False):
        super().__init__()
        # hidden layers
        self.hiddenSizes = hiddenSizes
        # self.dtype = dtype
        if any(hiddenSizes):
            inSizeH = inSize
            for i, l in enumerate(hiddenSizes):
                setattr(self, 'hidden_linear'+str(i), nn.Linear(inSizeH, l, bias = bias))
                inSizeH = l
            # output layer
            self.linear_out = nn.Linear(hiddenSizes[-1], outSize, dtype = dtype, bias = bias)
        else:
            self.linear_out = nn.Linear(inSize, outSize, dtype = dtype, bias = bias)
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = xb
        for i,_ in enumerate(self.hiddenSizes):
            out = getattr(self, 'hidden_linear%s' % i)(out)
        # Apply activation function
        # out = F.relu(out)

        # Get predictions using output layer
        out = self.linear_out(out)
        return out
    #
    # def training_step(self, batch):
    #     data, actual = batch
    #     out = self(data)
    #     return self.loss(out, actual)

    def validation_step(self, batch):
        data, actual = batch
        out = self(data)
        return {'val_loss': self.loss(out, actual), 'val_acc': accuracy(out, actual)}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))