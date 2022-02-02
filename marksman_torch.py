import io
import json
import torch
import numpy as np
import itertools as it
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from postgresql_db import SQLconn, SQLengine, composed_insert, composed_update, pg_execute


def get_default_device(device = 'cuda'):
    """Pick GPU if available, else CPU"""

    if (device.lower() == 'cuda' or device.lower() == 'gpu') and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Function that can move data and model to a chosen device.
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def save_io(s, *args, **kwargs):
    f = io.BytesIO()
    torch.save(s, f, *args, **kwargs)
    f.seek(0)
    return f


class DeviceDataLoader(tud.DataLoader):
    """Wrap a dataLoader to move data to a device"""
    def __init__(self, device, *args, **kwargs):
        self.device = device
        return super(DeviceDataLoader, self).__init__(*args, **kwargs)
    def __iter__(self):
        x = super(DeviceDataLoader, self).__iter__()
        return x
        # x, y = super(DeviceDataLoader, self).__iter__()
        # """Yield a batch of data after moving it to device"""
        # return (x.to(self.device), y.to(self.device))
    #
    # def __len__(self):
    #     """Number of batches"""
    #     return len(self.dl)


def param_tuples(model):
    return [(name[0] + name.split('.')[0].replace(name.split('.')[0].rstrip('0123456789'),'') + \
            '.' + ','.join(map(str,v)), p.data[v].item()) \
            for name,p in list(model.named_parameters()) \
            for v in it.product(*[range(i) for i in p.shape])]

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(model, epochs, trainLoader, testLoader, device, loss_fn = None, optimizer = None, metric_fn = None, dbPackage = None):
    """Train the model using gradient descent"""
    history = None
    result = test(model, testLoader, device = device, loss_fn = loss_fn, metric_fn = metric_fn)
    for epoch in range(epochs):
        if dbPackage is not None:
            v = dbPackage['values']
            zip_param = list(zip(*param_tuples(model)))
            col_dr = list(v.keys()) + ['epoch', "optimizer","binary", "binaryKeys", "json"] + list(zip_param[0])
            sqlComposed = composed_insert(dbPackage['table'], col_dr, schema = 'dt', returning = ['uuid'])
            vbn = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'optimizer' : optimizer}
            vjs = {'loss' : result['loss']}
            if metric_fn is not None: vjs['metric'] = metric
            val_dr = [tuple(v.values()) + (epoch, str(optimizer), save_io(vbn).read(), list(vbn.keys()).sort(), json.dumps(vjs)) + zip_param[1]]
            pg_execute(dbPackage['conn'], sqlComposed, val_dr, commit = True)

        train(model, trainLoader, device = device, loss_fn = loss_fn, optimizer = optimizer)
        result = test(model, testLoader, device = device, loss_fn = loss_fn, metric_fn = metric_fn)

        # model.epoch_end(epoch, result)
        if history is None:
            history = result.copy()
            history.update([(x, [y]) for x, y in result.items()])
        else:
            history.update([(x, history[x] + [y]) for x, y in result.items()])
        dbOut = {'loss' : result['loss']}

    return history, dbOut

def train(model, dataLoader, device = get_default_device(), loss_fn = None, optimizer = None, batchDisp = 500):
    """Train the model using gradient descent"""
    if loss_fn is None:
        loss_fn = model.loss
    if optimizer is None:
        optimizer = model.optimizer

    size = len(dataLoader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)
        # Compute prediction error
        pred = model(x)
        loss = model.loss(pred, y)
        # print(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('\n')
        # print(pred-y)
        # print(y)
        # print([p for p in model.parameters()])
        # print([p.grad for p in model.parameters()])
        # print('\nend')
        # print(2*(pred-y)*x)
        # if batchDisp is not None:
        #     if batch % batchDisp == 0:
        #         loss, current = loss.item(), batch * len(x)
        #         print([p for p in model.parameters()])
        #         print([p.grad for p in model.parameters()])
        #         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(model, dataLoader, device = get_default_device(), loss_fn = None, metric_fn = None):
    if loss_fn is None:
        loss_fn = model.loss
    if metric_fn is None and hasattr(model, 'metric'):
        metric_fn = model.metric

    loss, metric = 0, 0

    with torch.no_grad():
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            if metric_fn is not None:
                metric += metric_fn(pred, y)
        # print(y)
        # print(pred)
    loss /= len(dataLoader)
    # print([p for p in model.parameters()])
    # print([p.grad for p in model.parameters()])
    if metric_fn is not None:
        metric /= len(dataLoader.dataset)
        # print(f"Test Error: \n {metric('', '')}: {(metric):>0.1f}%, Avg loss: {loss:>8f} \n")
        return {'loss' : loss, 'metric' : metric}
    else:
        # print(f"Test Error: \n Avg loss: {loss:>8f} \n")
        return {'loss' : loss}
