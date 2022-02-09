from ib_insync import *
from marksman_db import *
from marksman_torch import *
from marksman_models import *
from marksman_objects import *
from postgresql_db import SQLconn, SQLengine, composed_insert, composed_update, pg_execute
# from marksman_models import *
from datetime import datetime, timedelta, date
import math as m
import numpy as np
import pandas as pd
import itertools as it
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.utils.data as tud
import torch.nn.functional as F
import json
import io
import itertools as it
from uuid import uuid4


matplotlib.rcParams['figure.facecolor'] = '#ffffff'
commit = True
uploadDS = False
ds_uuid = 'b43b553a-3427-4acb-ba54-a6178025cad5'# data set uuid
ds_uuid = '90cf0263-4460-4720-b944-3251460c5a19'
# ds_uuid = ''

dm_uuid = '3ed6c3a9-5fc0-432e-8b97-715745c73bb2'
dm_uuid = ''

# Stock data
tickers = ['AMZN','AAPL']
barSizesSamples = {'1 month': 5 , '1 week': 5 , '1 day': 5}
barSizesSamples = {'1 month': 1}
inputColumns = ['percent']
targetColumns = ['percent']
startDate = datetime(2010,7,1)
endDate = None
index = 'date'
toJSON = {'tickers': tickers, 'barSizesSamples': barSizesSamples, 'inputColumns': inputColumns,
            'targetColumns': targetColumns, 'startDate': startDate, 'endDate': endDate}
# aggregate_tables(SQLconn, ['AMZN_1 day', 'AAPL_1 day', 'AAPL_1 month'], inputColumns, 'date', startDate = startDate, endDate = endDate, CREATE = True)
# Model params
hiddenSizes = [3,5,8]
outSize = len(targetColumns)*len(tickers)
# Training params
batchSize = 1
testFrac = 0.3
if ds_uuid:
    inputs, targets, inputNames, targetNames, tickers, columns = fetch_ds(SQLconn, ds_uuid)
else:
    inputs, targets, inputNames, targetNames, ds_uuid = build_data(SQLconn, tickers, barSizesSamples, inputColumns, targetColumns, schema = 'trades',
                                                                    startDate = startDate, endDate = endDate, UPLOAD = uploadDS)
print(inputNames)
print(targetNames)
print(inputs.shape)
# [print('{}: {}'.format(e,p)) for e,p in zip(inputNames, inputs[0])]


cov = np.cov(inputs.transpose())
mean = inputs.mean(axis=0)
absMean = abs(inputs).mean(axis=0)
span = inputs.max(axis=0)-inputs.min(axis=0)
median = np.median(inputs)


dtype = torch.float
inputs = torch.from_numpy(inputs).type(dtype)
targets = torch.from_numpy(targets).type(dtype)
DS = tud.TensorDataset(inputs, targets)

model = MultiLayerModel1(inputs.shape[-1], hiddenSizes, outSize)
if not dm_uuid and commit:
    torchBinary = {'model': model}
    col_dm = ['tickers', 'columns', 'index', 'text', 'binary', 'binaryKeys', 'ds_uuid']
    sqlComposed = composed_insert('models_prime', col_dm, schema = 'dm', returning = ['uuid'])
    val_dm = [(tickers, columns, index, str(torchBinary).replace('\n', ''), save_io(torchBinary).read(), list(torchBinary.keys()).sort(), ds_uuid)]
    dm_uuid, = pg_execute(SQLconn, sqlComposed, val_dm, commit = True)[0]
# Parameters = [p for p in list(model.named_parameters())]

if commit:

    col_dr = ['tickers', 'columns', 'index', 'dm_uuid', 'ds_uuid']
    dr_uuid, = pg_execute(SQLconn, composed_insert('runs_prime', col_dr, schema = 'dr', returning = ['uuid']),
                            [(tickers, columns, index, dm_uuid, ds_uuid)], commit = True)[0]

    constraint = [[dr_uuid + '_pkey', '$PRIMARY KEY$', ('uuid',)],
                    [dr_uuid + '_fkeyr', '$FOREIGN KEY$', ('dr_uuid',), '$REFERENCES$','dr','$.$','runs_prime', ('uuid',)],
                    [dr_uuid + '_fkeym', '$FOREIGN KEY$', ('dm_uuid',), '$REFERENCES$', 'dm','$.$','models_prime', ('uuid',)]]
    col = (['loss', '$double precision$'], ['epoch', '$double precision$', '$NOT NULL$'],
                    ['epochs', '$double precision$'], ['optimizer', '$text$'],
                    ['dt_uuid', '$uuid$', '$NOT NULL$', '$REFERENCES$', 'dt','$.$','training_prime', '$ON DELETE CASCADE$'])

    ColumnLists = tuple(map(list, it.product(list(zip(*param_tuples(model)))[0], ['$double precision$'])))

    cmp_create = composed_create(dr_uuid,  col + ColumnLists, schema = 'dt', inherits = 'training_prime', constraint = constraint)
    pg_execute(SQLconn, cmp_create,commit = True)

optimizerf = torch.optim.SGD
optimizer = optimizerf(model.parameters(), 0.1)

for u in range(0,1):
    model = MultiLayerModel1(inputs.shape[-1], hiddenSizes, outSize)
    trainDS, testDS = tud.random_split(DS, [m.floor((1-testFrac)*len(DS)), m.ceil(testFrac*len(DS))])
    trainDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
    testDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
    device = get_default_device('cpu')
    # trainDL = DeviceDataLoader(device, trainDS, batchSize, shuffle=True)
    # print(inputs.mean())
    # print(inputs.std())

    # model.to(device = device).to(device = device)
    optimizerf = torch.optim.SGD
    lr = 0.0000003
    epochs = 1
    optimizer = optimizerf(model.parameters(), lr)
    # print(model.state_dict())
    # print(model)
    # torch.save(model, r"C:\Users\user\Desktop\temp\lol.json")
    # print(torch.load(r"C:\Users\user\Desktop\temp\lol.json"))
    # epochs = 5
    # lr = 30
    # print(test(model, testDL, device = device))
    # print('\nBEFORE:')
    # print([p for p in model.parameters()])
    # dbPackage = None
    if commit:
        vin = {'tickers': tickers, 'columns':  columns, 'index': index, 'dr_uuid': dr_uuid, 'dm_uuid': dm_uuid, 'ds_uuid': ds_uuid}
        sqlC = composed_insert('training_prime', list(vin.keys()) + ['binary', 'binaryKeys', 'json'], schema = 'dt', returning = ['uuid'])
        vjs = {'model': str(model), 'optimizer': str(optimizer), 'epochs': epochs}
        vbn = {'model': model, 'model_state_dict': model.state_dict(),  'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf}
        vpg = [tuple(vin.values()) + (save_io(vbn).read(), list(vbn.keys()).sort(), json.dumps(vjs, sort_keys=True, default=str))]

        dt_uuid, = pg_execute(SQLconn, sqlC, vpg, commit = True)[0]

        dbPackage = {'conn': SQLconn, 'table': dr_uuid, 'commit': commit, \
                    'values': {'dt_uuid': dt_uuid, 'epochs': epochs} | vin}
    history, dbFit = fit(model, epochs, trainDL, testDL, device, optimizer = optimizer, dbPackage = dbPackage)

    if commit:
        where = ['uuid', '$=$', '%s']
        whereVal = [dt_uuid]
        col_dt = ['binary', 'binaryKeys', ['json', 'json', '$||$', '%s']]
        sqlC = composed_update('training_prime', col_dt, schema = 'dt', returning = ['uuid'], where = where)
        vbn = {'model': model, 'model_state_dict': model.state_dict(),  'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf}
        ColumnLists = tuple(map(list, it.product(list(zip(*param_tuples(model)))[0], ['$double precision$'])))
        vpg = [save_io(vbn).read(), list(vbn.keys()).sort(), json.dumps(dbFit, sort_keys=True, default=str)]

        pg_execute(SQLconn, sqlC, vpg + whereVal, commit = True)
    # print(out)
    # print('AFTER:')
    # print([p for p in model.named_parameters()])
    # for p in model.parameters():
    #     print(p.cpu().detach().numpy())
    #     break
# #
# # for input, target in trainDL:
# #     outputs = model(input)
# #     print(outputs)
# #     loss = model.loss(outputs, target)
# #     print('Loss:', loss.item())
