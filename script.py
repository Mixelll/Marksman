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
def dumps(j): return json.dumps(j, sort_keys=True, default=str)
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
                                                                    startDate=startDate, endDate=endDate, UPLOAD=uploadDS)
print(inputNames)
print(targetNames)
print(type(inputs))
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
    col = ['tickers', 'columns', 'index', 'text', 'binary', 'binaryKeys', 'ds_uuid']
    val = [(tickers, columns, index, str(torchBinary).replace('\n', ''), save_io(torchBinary).read(),
            list(torchBinary.keys()).sort(), ds_uuid)]
    cmp_insert = composed_insert('models_prime', col, schema='dm', returning=['uuid'])
    dm_uuid, = pg_execute(SQLconn, cmp_insert, val, commit=True)[0]
# Parameters = [p for p in list(model.named_parameters())]

if commit:
    col = ['tickers', 'columns', 'index', 'dm_uuid', 'ds_uuid']
    val = [(tickers, columns, index, dm_uuid, ds_uuid)]
    cmp_insert = composed_insert('runs_prime', col, schema='dr', returning=['uuid'])
    dr_uuid, = pg_execute(SQLconn, cmp_insert, val, commit=True)[0]
    col = (['dt_uuid', '$uuid$', '$NOT NULL$', '$REFERENCES$', 'dt','$.$','training_prime', '$ON DELETE CASCADE$'],)
    ColumnLists = tuple(map(list, it.product(list(zip(*param_tuples(model)))[0], ['$double precision$'])))
    cmp_create = composed_create(dr_uuid,  col + ColumnLists, schema='dt', like='training_prime')
    pg_execute(SQLconn, cmp_create, commit=True)
    drStartDate = datetime.now()

optimizerf = torch.optim.SGD
optimizer = optimizerf(model.parameters(), 0.1)

for u in range(0,1):
    model = MultiLayerModel1(inputs.shape[-1], hiddenSizes, outSize)
    trainDS, testDS = tud.random_split(DS, [m.floor((1-testFrac)*len(DS)), m.ceil(testFrac*len(DS))])
    trainDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
    testDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
    device = get_default_device('cpu')
    optimizerf = torch.optim.SGD
    lr = 0.0000003
    epochs = 1
    optimizer = optimizerf(model.parameters(), lr)

    if commit:
        vin = {'tickers': tickers, 'columns':  columns, 'index': index, 'dr_uuid': dr_uuid, 'dm_uuid': dm_uuid, 'ds_uuid': ds_uuid}
        col = list(vin.keys()) + ['binary', 'binaryKeys', 'json_hr', 'json']
        dictJSONhr = {'epochs': epochs, 'testFrac' : testFrac}
        dictJSON = {'model': str(model), 'optimizer': str(optimizer)} | dictJSONhr
        vbn = {'model': model, 'model_state_dict': model.state_dict(), 'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf}
        val = [tuple(vin.values()) + (save_io(vbn).read(), list(vbn.keys()).sort(), dumps(dictJSONhr), dumps(dictJSON))]
        cmp_insert = composed_insert('training_prime', col, schema='dt', returning=['uuid'])
        dt_uuid, = pg_execute(SQLconn, cmp_insert, val, commit=True)[0]

        dbPackage = {'conn': SQLconn, 'table': dr_uuid, 'commit': commit, \
                    'values': {'dt_uuid': dt_uuid, 'epochs': epochs} | vin}
        dtStartDate = datetime.now()

    history, fitDict = fit(model, epochs, trainDL, testDL, device, optimizer=optimizer, dbPackage=dbPackage)

    if commit:
        dtEndDate = datetime.now()

        col = ['binary', 'binaryKeys', ['json_hr', 'json_hr', '$||$', '%s'], ['json', 'json', '$||$', '%s']]
        vbn = {'model': model, 'model_state_dict': model.state_dict(),  'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf}
        ColumnLists = tuple(map(list, it.product(list(zip(*param_tuples(model)))[0], ['$double precision$'])))
        dictJSONhr = fitDict | {'dt_time' : (dtEndDate - dtStartDate).total_seconds()}
        val = [save_io(vbn).read(), list(vbn.keys()).sort(), dumps(dictJSONhr), dumps(dictJSONhr)]
        whereVal = dt_uuid
        where = ['uuid', '$=$', '%s']
        cmp_update = composed_update('training_prime', col, schema='dt', returning=['uuid'], where=where)
        pg_execute(SQLconn, cmp_update, val + [whereVal], commit=True)


if commit:
    drEndDate = datetime.now()
    col = ['endDate']
    val = [drEndDate]
    whereVal = dr_uuid
    where = ['uuid', '$=$', '%s']
    cmp_update = composed_update('runs_prime', col, schema='dr', where=where)
    pg_execute(SQLconn, cmp_update, val + [whereVal], commit=True)
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
