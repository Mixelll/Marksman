import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud

import marksman_db as mdb
import marksman_extras as me
import marksman_torch as mtor
import postgresql_db as db

from math import floor, ceil
from uuid import uuid4
from datetime import datetime, timedelta, date
from itertools import product

from marksman_models import MultiLayerModel1
from postgresql_db import SQLconn as SQLconn


def dumps(j): return json.dumps(j, sort_keys=True, default=str)

# Configure run
traningIter = 2  # Number of same-parameter iterations, the sole delta is shuffled data
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

# Model params
hiddenSizes = [3,5,8]

# Training params
batchSize = 1
testFrac = 0.3

# Fetch or build data
if ds_uuid:
    inputs, targets, inputNames, targetNames, tickers, columns = mdb.fetch_ds(SQLconn, ds_uuid)
else:
    inputs, targets, inputNames, targetNames, ds_uuid = mdb.build_data(SQLconn, tickers, barSizesSamples, inputColumns, targetColumns, schema = 'trades',
                                                                    startDate=startDate, endDate=endDate, UPLOAD=uploadDS)
print(inputNames)
print(targetNames)
print(inputs.shape)
# [print('{}: {}'.format(e,p)) for e,p in zip(inputNames, inputs[0])]

# Some statistics
# cov = np.cov(inputs.transpose())
# mean = inputs.mean(axis=0)
# absMean = abs(inputs).mean(axis=0)
# span = inputs.max(axis=0)-inputs.min(axis=0)
# median = np.median(inputs)

# Cast data and deduce model outer layer size - outSize
dtype = torch.float
inputs = torch.from_numpy(inputs).type(dtype)
targets = torch.from_numpy(targets).type(dtype)
DS = tud.TensorDataset(inputs, targets)
outSize = len(targetColumns)*len(tickers)

# Fetch or build model
model = MultiLayerModel1(inputs.shape[-1], hiddenSizes, outSize)
# Parameters = [p for p in list(model.named_parameters())]

# insert model to DB table dm.models_prime
if not dm_uuid and commit:
    torchBinary = {'model': model}
    col = ['tickers', 'columns', 'index', 'text', 'binary', 'binaryKeys', 'ds_uuid']  # table columns
    val = [(tickers, columns, index, str(torchBinary).replace('\n', ''), mtor.save_io(torchBinary).read(),
            list(torchBinary.keys()).sort(), ds_uuid)]  # insert values
    cmp_insert = db.composed_insert('models_prime', col, schema='dm', returning=['uuid'])  # query without values
    dm_uuid, = db.pg_execute(SQLconn, cmp_insert, val, commit=True)[0] # parse values to query and execute


# insert run start to 'runs_prime' and create {dr_uuid} run table
if commit:
    # upload run start to DB table dr.runs_prime
    col = ['tickers', 'columns', 'index', 'dm_uuid', 'ds_uuid']
    val = [(tickers, columns, index, dm_uuid, ds_uuid)]
    cmp_insert = db.composed_insert('runs_prime', col, schema='dr', returning=['uuid'])
    dr_uuid, = db.pg_execute(SQLconn, cmp_insert, val, commit=True)[0]
    # create run table on DB as dr."dr_uuid"
    col = (['dt_uuid', '$uuid$', '$NOT NULL$', '$REFERENCES$', 'dt','$.$','training_prime', '$ON DELETE CASCADE$'],)
    ColumnLists = tuple(map(list, product(list(zip(*mtor.param_tuples(model)))[0], ['$double precision$'])))
    cmp_create = db.composed_create(dr_uuid,  col + ColumnLists, schema='dt', like='training_prime')
    db.pg_execute(SQLconn, cmp_create, commit=True)
    drStartDate = datetime.now()
else:
    dr_uuid = uuid4()

optimizerf = torch.optim.SGD

# Iterate over hyperparameters
epochsL = [5]
lrL = [1e-4, 1e-3, 1e-2, 1e-1]

print(f'RUN {dr_uuid} START')
print(f'MODEL: {model}')
print(f'OPTIMIZER FUNCTION: {optimizerf}')
totalTuples = me.iter_length(product(epochsL, lrL))

for i, pr in enumerate(product(epochsL, lrL)):
    epochs = pr[0]
    optimizer = optimizerf(model.parameters(), pr[1])
    print(f'Starting hyperparam tuple #{epochsL} OF {totalTuples} FOR {epochsL} EPOCHS ')
    print(f'Optimizer: {optimizer}')

    # train with shuffle arrays, traningIter times
    for u in range(0, traningIter):
        print(f'Iter #{u} start')
        trainDS, testDS = tud.random_split(DS, [floor((1-testFrac)*len(DS)), ceil(testFrac*len(DS))])
        trainDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
        testDL = tud.DataLoader(trainDS, batchSize, shuffle=True)
        device = mtor.get_default_device('cpu')

        # upload training start to DB table dt.training_prime
        if commit:
            vin = {'tickers': tickers, 'columns':  columns, 'index': index,
                    'dr_uuid': dr_uuid, 'dm_uuid': dm_uuid, 'ds_uuid': ds_uuid}  # column-values pairs
            col = ['json_hr', 'json', 'binary', 'binaryKeys'] + list(vin.keys())
            jshr = {'epochs': epochs, 'testFrac' : testFrac}  # to-JSON dict intended to be human readable
            js = {'model': str(model), 'optimizer': str(optimizer)} | jshr  # main to-JSON dict
            bn = {'model': model, 'model_state_dict': model.state_dict(), 'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf} # to-binary dict
            val = [(dumps(jshr), dumps(js), mtor.save_io(bn).read(), list(bn.keys()).sort()) + tuple(vin.values())]
            cmp_insert = db.composed_insert('training_prime', col, schema='dt', returning=['uuid'])
            dt_uuid, = db.pg_execute(SQLconn, cmp_insert, val, commit=True)[0]

            dbPackage = {'conn': SQLconn, 'table': dr_uuid, 'commit': commit, \
                        'values': {'dt_uuid': dt_uuid} | vin}
            dtStartDate = datetime.now()

        # train model and calculate loss per epoch, the fit function uploads
        history, fitDict = mtor.fit(model, epochs, trainDL, testDL, device, optimizer=optimizer, dbPackage=dbPackage)

        # update training end in DB table dt.training_prime
        if commit:
            dtEndDate = datetime.now()

            col = [['json_hr', 'json_hr', '$||$', '%s'], ['json', 'json', '$||$', '%s'], 'binary', 'binaryKeys', ]
            bn = {'model': model, 'model_state_dict': model.state_dict(),  'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(), 'optimizerf': optimizerf}
            ColumnLists = tuple(map(list, product(list(zip(*mtor.param_tuples(model)))[0], ['$double precision$'])))
            jshr = fitDict | {'dt_time' : (dtEndDate - dtStartDate).total_seconds()}
            jsh = jshr
            val = [dumps(jshr), dumps(js), mtor.save_io(bn).read(), list(bn.keys()).sort()]
            whereVal = dt_uuid
            where = ['uuid', '$=$', '%s']
            cmp_update = db.composed_update('training_prime', col, schema='dt', returning=['uuid'], where=where)
        db.pg_execute(SQLconn, cmp_update, val + [whereVal], commit=True)
        print(f'Iter #{u} end')

# update run end in DB table dr.runs_prime
if commit:
    drEndDate = datetime.now()
    col = ['endDate']
    val = [drEndDate]
    whereVal = dr_uuid
    where = ['uuid', '$=$', '%s']
    cmp_update = db.composed_update('runs_prime', col, schema='dr', where=where)
    db.pg_execute(SQLconn, cmp_update, val + [whereVal], commit=True)
print(f'RUN #{dr_uuid} END')
