from postgresql_db import *
from marksman_extras import *
from marksman_objects import *
import json
import numpy as np
import pandas as pd
import itertools as it
from psycopg2 import sql
from datetime import datetime, timedelta, date

def df_insert_prime(conn, tblName, df, schema = None):
    s, n, p = operators()
    d = lambda o : df_ind_col_op(df, 'date', o)
    primeName = schema + '_prime'
    nf = {'name' : tblName, 'ticker' : tblName.split('_')[0], 'barSize' : tblName.split('_')[1],
                'columns' : [], 'startDate' : d(min), 'endDate' : d(max),
                'json' : json.dumps('')} #df.index.names + list(df.columns.array)
    conflict = 'name'
    print(stat_dic(df))
    print(json.dumps(stat_dic(df)))
    qn = ['startDate', 'endDate']
    q = s(' SET {s} = least({s}, %s), {e} = greatest({e}, %s)').format(s=n(qn[0]), e=n(qn[1]))
    set = [['startDate', '$least($', ([primeName, 'startDate'],), '$,$', '%s)'], ['endDate', '$greatest($', ([primeName, 'endDate'],), '$,$', '%s)']]
    insert_prime = composed_insert(primeName, list(nf.keys()), schema=schema, conflict=conflict, set=set)
    print(insert_prime)
    pg_execute(conn, insert_prime, [tuple(nf.values()), nf[qn[0]], nf[qn[1]]])

def join_indexed_tables(conn, tableNames, columns, index, join=None,  schema=None, startDate=None, endDate=None):
    n = sql.Identifier
    if not isinstance(schema, list): schema = [schema]*len(tableNames)

    comp = [sql.SQL('SELECT {}.{}, {} ').format(n(tableNames[0]), n(index), composed_columns(it.product(tableNames, columns), AS=True))]
    comp.append(composed_from_join(tables=zip(schema, tableNames), using=[index]*len(tableNames), join=join))

    cp, execV = composed_between(start = startDate, end = endDate)
    comp.extend([sql.SQL('WHERE {} ').format(n(index)), cp])
    return pd.read_sql_query(conn.cursor().mogrify(sql.Composed(comp), execV), conn)

def build_data(conn_or_engine, tickers, barSizesSamples, inputColumns, targetColumns,
                schema=None, UPLOAD=False, order='F', startDate=None, endDate=None, **kwargs):
    barSizes = list(barSizesSamples.keys())
    barSizes = [x for _, x in sorted(zip(td_parser(barSizes), barSizes), key=lambda pair: pair[0])]
    table_names = lambda x : [e + '_' + x for e in tickers]
    columns = inputColumns[:]

    for s in targetColumns:
        if s not in columns:
            columns.append(s)
    dfs = {}
    for x in barSizes:
        dfs[x]  = join_indexed_tables(conn_or_engine, table_names(x), columns, 'date', schema = schema, startDate = startDate, endDate = endDate, **kwargs)
        print(dfs[x])

    bs = barSizes[0]
    xs = barSizesSamples[bs]
    targetColumnsBar = ['.'.join(e) for e in it.product(table_names(barSizes[0]), targetColumns)]
    inputColumnsBar0 = ['.'.join(e) for e in it.product(table_names(barSizes[0]), inputColumns)]
    if order == 'C':
        inputColumnsBar = ['.'.join(e[1:3]) + str(e[0]) for x in barSizes for e in it.product(reversed(range(0,barSizesSamples[x])), table_names(x), inputColumns)]
    elif order == 'F':
        inputColumnsBar = ['.'.join(e[0:2]) + str(e[2]) for x in barSizes for e in it.product(table_names(x), inputColumns, reversed(range(0,barSizesSamples[x])))]
    print(inputColumnsBar)
    target = np.full([dfs[bs].shape[0], len(targetColumnsBar)], np.NaN)
    input = np.full([dfs[bs].shape[0], len(inputColumnsBar0) * xs], np.NaN)
    for i in reversed(range(xs, dfs[bs].shape[0])):
        date = dfs[bs].loc[i, 'date']
        input[i,:] = np.ravel(dfs[bs].loc[i-xs:i-1, inputColumnsBar0].to_numpy(), order = order)
        # print(date)
        # print(input[i,:])
        target[i,:] = dfs[bs].loc[i, targetColumnsBar]
    ColEx = {}
    if len(barSizes)>1:
        index, elemN , is_break= {}, 0, False
        for x in barSizes[1:]:
            xs = barSizesSamples[x]
            ColEx[x] = ['.'.join(e) for e in it.product(table_names(x), inputColumns)]
            index[x] = dfs[x].shape[0]-1
            elemN += len(ColEx[x])*xs
        extraInput = np.full([dfs[bs].shape[0], elemN], np.NaN)
        for i in reversed(range(xs, dfs[bs].shape[0])):
            dEx  = np.array([])
            date = dfs[bs].loc[i, 'date']
            for x in barSizes[1:]:
                xs = barSizesSamples[x]
                while True:
                    if dfs[x].loc[index[x], 'date'] <= date:
                        dEx = np.append(dEx, np.ravel(dfs[x].loc[index[x]-xs+1:index[x], ColEx[x]].to_numpy(), order = order))
                        break
                    else:
                        index[x] -= 1
                        if index[x] < xs:
                            is_break = True
                            break
            if not is_break:
                extraInput[i,:] = dEx
            else:
                break
        min_i = np.amax(np.where(np.apply_along_axis(np.isnan, 1, extraInput))[0]) +1
        input = np.concatenate((input, extraInput), axis=1)
    else:
        min_i = np.amax(np.where(np.apply_along_axis(np.isnan, 1, input))[0]) +1
    columnsAll = ['i.'+ x for x in inputColumnsBar] + ['t.'+ x for x in targetColumnsBar]
    if UPLOAD:
        # if startDate is not None:
        toJSON = {'tickers' : tickers, 'barSizesSamples' : barSizesSamples, 'inputColumns' : inputColumns, \
        'targetColumns' : targetColumns, 'startDate' : startDate, 'endDate' : endDate}
        df = pd.DataFrame(np.concatenate((input[min_i: , :], target[min_i: , :]), axis=1), columns = columnsAll)
        df.insert(0, 'date', pd.to_datetime(dfs[barSizes[0]].loc[min_i:,'date'].reset_index()['date'], utc = True)) # .map(lambda x : pd.to_datetime(x))
        sqlComposed = composed_insert('datasets_prime', ["tickers","columns","index","startDate", "endDate", "json"], schema = 'ds', returning = ['uuid'])
        uuid = pg_execute(SQLconn, sqlComposed, (tickers, columnsAll, 'date', df['date'].min(), df['date'].max(), json.dumps(toJSON, sort_keys=True, default=str)), commit = True)[0][0]
        append_df_to_db(SQLengine, uuid, df, schema = 'ds')
    else:
        uuid = None
    return input[min_i: , :], target[min_i: , :], inputColumnsBar, targetColumnsBar, uuid

def fetch_ds(SQLconn, uuid):
    tickers, columns = pg_execute(SQLconn, sql.SQL("SELECT {},{} FROM ds.datasets_prime WHERE uuid = %s").format(sql.Identifier('tickers'), sql.Identifier('columns')), values = [uuid])[0]
    df = get_table_as_df(SQLconn, uuid, schema = 'ds')
    columns = df.columns.tolist()
    print(columns)
    columns.remove('date')
    inputNames, targetNames, inputNamesO, targetNamesO = [], [], [], []
    for x in columns:
        xs =  x.split('.')
        if xs[0] == 'i':
            inputNames.append('.'.join(xs[1:]))
            inputNamesO.append(x)
        elif xs[0] == 't':
            targetNames.append('.'.join(xs[1:]))
            targetNamesO.append(x)
    return df[inputNamesO].to_numpy(), df[targetNamesO].to_numpy(), inputNames, targetNames, tickers, columns
