# from inspect import getmembers, isfunction, isclass
import psycopg2 as psg
from psycopg2 import sql
import pandas as pd
from sqlalchemy import create_engine
import io
from marksman_extras import *
import uuid
from datetime import datetime, timedelta, date

with open('connections.txt') as f:
    exec(f.read())
SQLconn
SQLengine


def get_table_as_df(connOrEngine, tbl, columns=None, schema=None, **kwargs):
    return query_get_df(connOrEngine, sql.SQL('SELECT {} FROM {}{}') \
        .format(composed_columns(columns), composed_dot(schema), sql.Identifier(tbl)), **kwargs)


def query_get_df(connOrEngine, query):
    try:
        return pd.read_sql_query(query, con=connOrEngine)
    except:
        cur = connOrEngine.cursor()
        cur.execute(query)
        return pd.DataFrame(cur.fetchall())


def pg_execute(conn, query, values=None, commit=True):
    cur = conn.cursor()
    # print(query.as_string(conn))
    print(cur.mogrify(query, values))
    cur.execute(query, values)
    if commit:
        conn.commit()
    if cur.description is not None:
        return cur.fetchall()


def composed_parse(exp, enclose=False, parse=None):
    s, n, p = operators()
    if isinstance(exp, str):
        if exp[0]=='$' and exp[-1]=='$':
            if exp.strip('$ []()').lower() in ['primary key', 'foreign key', 'references', 'on delete cascade','on delete restrict',
                                    'unique', 'not null', 'default', 'current_timestamp', 'timestamp with time zone', 'text',
                                    'uuid', 'double precision', 'json', 'bytea', 'timestamp', 'like', 'and', 'or', 'join',
                                    'inner', 'left', 'right', 'full', '','.', ',', '=', '||', '(', ')', 'min', 'max', 'least', 'greatest']:
                returned =  s(exp.strip('$'))
            elif exp.strip('$ []()') == '%s':
                e = exp.strip('$').split('%s')
                returned = s(e[0]) + p + s(e[1])
            else:
                raise ValueError('Expression: ' + exp.strip('$ []()') + ' not found in approved expressions')
        elif exp.strip('$ []()') == '%s':
            e = exp.split('%s')
            returned = s(e[0]) + p + s(e[1])
        else:
            returned =  n(exp.strip('$'))
    elif isinstance(exp, dict):
        returned = composed_columns(filter(mbool, [k for k in exp.keys() if exp[k]]), enclose=enclose, parse=parse)
    elif isinstance(exp, tuple):
        returned = composed_columns(filter(mbool, exp), enclose=enclose, parse=parse)
    elif isinstance(exp, sql.Placeholder):
        returned = exp
    else:
        expPrev = exp[0]
        for x in exp[1:]:
            if x == expPrev:
                raise ValueError("Something's funny going on - a pattern is repeated")
            else:
                expPrev = x

        return sql.Composed([composed_parse(x, enclose) for x in filter(mbool, exp)])

    return s(' ') + returned + s(' ')


def composed_insert(tbl, columns, schema=None, returning=None, conflict=None, nothing=False, set=None):
    s, n, p = operators()
    comp = s('INSERT INTO {}{} ({}) VALUES {} ').format(composed_dot(schema), n(tbl), composed_separated(columns), p)

    if conflict:
        if nothing:
            comp += s('ON CONFLICT ({}) DO NOTHING').format(n(conflict))
        else:
            if set is None:
                set = columns
            comp += s('ON CONFLICT ({}) DO UPDATE').format(n(conflict)) + composed_set(set)

    if returning is not None:
        comp += s(' RETURNING {}').format(composed_separated(returning))

    return comp


def composed_update(tbl, columns, returning=None, schema=None, where=None):
    s, n, p = operators()
    comp = s('UPDATE {}').format(composed_dot(schema))
    comp += n(tbl) + composed_set(columns)

    if where is not None:
        comp += s(' WHERE ' ) + composed_parse(where)

    if returning is not None:
        comp += s(' RETURNING {}').format(composed_separated(returning))

    return comp


def composed_create(tbl, columns, schema=None, schema2=None, like=None, inherits=None, constraint=None):
    s, n, p = operators()
    if schema2 is None: schema2 = schema
    if isinstance(columns[0], str):
        columns = [columns]
    schema2 = composed_dot(schema2)
    comp = s('CREATE TABLE {}{} (').format(composed_dot(schema), n(tbl))
    if like is not None:
        comp += s('LIKE {}{} INCLUDING ALL,').format(schema2, n(like))
    comp += composed_parse(columns, parse=composed_parse)
    if constraint:
        if isinstance(constraint[0], str): constraint = [constraint]
        comp += s(', ')
        for c in constraint[:-1]:
            comp += s(' CONSTRAINT ') + composed_parse(c, enclose=True) + s(',')
        comp += s(' CONSTRAINT ') + composed_parse(constraint[-1], enclose=True)
    comp += s(') ')

    if inherits is not None:
        comp += s('INHERITS ({}{})').format(schema2, n(inherits))

    return comp


def composed_select_from_table(tbl, columns=None, schema=None):
    return sql.SQL('SELECT {} FROM {}{} ').format(composed_columns(columns), composed_dot(schema), sql.Identifier(tbl))


def composed_from_join(join=None, tables=None, columns=None, using=None):
    s, _, _ = operators()
    def n(x): composed_separated(x, '.')
    joinc = []
    for v in multiply_iter(join, max(iter_length(tables, columns, using))):
        vj = '$'+v+'$' if v else v
        joinc.append(composed_parse([vj, '$ JOIN $']))

    if tables:
        tables = list(tables)
        comp = s('FROM {} ').format(n(tables[0]))
        if using:
                for t, u, jo in zip(tables[1:], using, joinc):
                    comp += jo + s('{} USING ({}) ').format(n(t), composed_columns(u))
        elif columns:
            for t, co, jo in zip(tables[1:], columns, joinc):
                comp += jo + s('{} ').format(n(t))
                for j, c in enumerate(co):
                    comp += s('ON {} = {} ').format(n(c[0]), n(c[1]))
                    if j < len(co): comp += s('AND ')
        else:
            comp += s('NATURAL ') + composed_parse([join, '$ JOIN $']) + s('{} ').format(n(table[i]))
    elif columns:
        columns = list(columns)
        comp = s('FROM {} ').format(n(columns[0][:-1]))
        for i in range(1, len(columns)):
            comp += joinc[i-1] + s('{} ON {} = {} ').format(*map(n,[columns[i][:-1], columns[i-1], columns[i-1]]))
    else:
        raise ValueError("Either tables or columns need to be given")

    return comp


def composed_set(columns):
    s, n, p = operators()
    if not columns:
        return s('')

    col = [];
    val = [];
    for c in columns:
        if isinstance(c, (tuple, list)):
            if len(c)>1:
                col.append(c[0])
                val.append(c[1:])
            else:
                col.append(c[0])
                val.append(p)
        else:
                col.append(c)
                val.append(p)
    if len(col)>1:
        formatted = s(' SET ({}) = ({})')
    else:
        formatted = s(' SET {} = {}')
    return formatted.format(composed_columns(col), composed_separated(val, parse=composed_parse))


def composed_between(start=None, end=None):
    s = operators()[0]
    comp = s('')
    execV = []

    if start is not None and end is not None:
        comp += s('BETWEEN %s AND %s ')
        execV.extend([start, end])
    elif start is None and end is not None:
        comp += s('<= %s ')
        execV.append(end)
    elif start is not None:
        comp += s('>= %s ')
        execV.append(start)

    return comp, execV


def composed_dot(name):
    s, n, _ = operators()
    if name:
        if not isinstance(name, str):
            return [composed_dot(x) for x in name]
        return s('{}.').format(n(name))
    return s('')



def composed_columns(columns, enclose=False, parse=None, **kwargs):
    s = operators()[0]
    if parse is None:
        parse = lambda x: composed_separated(x, '.', **kwargs)
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        return s('*')
    else:
        comp = s(', ').join(map(parse, columns))
        if enclose:
            return s('(') + comp + s(')')
        return comp


def composed_separated(names, sep=', ', AS=False, parse=None):
    s, n, _ = operators()
    if not parse:
        parse = n
    if isinstance(names, str):
        names = [names]
    names = list(filter(mbool, names))

    if sep in [',', '.', ', ', ' ', '    ']:
        comp = s(sep).join(map(parse, names))
        if AS:
            comp += s(' ') + n(sep.join(names))
        return comp
    else:
        raise ValueError('Expression: "' + sep + '" not found in approved separators')


def append_df_to_db(engine, tbl, df, schema=None, index=True):
    conn = engine.raw_connection()
    df.head(0).to_sql(tbl, engine, if_exists='append', index=index, schema=schema)
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=index)
    output.seek(0)

    if schema:
        cur.copy_from(output,  tbl, null="") # null values become ''
    else:
        cur.copy_expert(sql.SQL("COPY {}.{} FROM STDIN DELIMITER '\t' CSV HEADER;") \
            .format(sql.Identifier(schema), sql.Identifier(tbl)), output)
    conn.commit()


def upsert_df_to_db(engine, tbl, df, schema=None, index=True):
    s, n, _ = operators()
    conn = engine.raw_connection()
    df.head(0).to_sql(tbl, engine, if_exists='replace', index=index, schema=schema)
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=index)
    output.seek(0)

    temp, schema, tbl = n(tbl + '_' + str(uuid.uuid4())[:8]), composed_dot(schema), n(tbl)
    cur.execute(s('CREATE TEMP TABLE {} (LIKE {}{} INCLUDING ALL);').format(temp, schema, tbl))
    cur.copy_expert(s("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(temp), output)
    cur.execute(s('DELETE FROM {}{} WHERE ({index}) IN (SELECT {index} FROM {});')
        .format(schema, tbl, temp, index=composed_parse(tuple(df.index.names))))
    cur.execute(s('INSERT INTO {}{} SELECT * FROM {};').format(schema, tbl, temp))
    cur.execute(s('DROP TABLE {};').format(temp))
    conn.commit()


def get_tableNames(conn, names, operator='like', not_=False, relkind=('r', 'v'), case=False, schema=None, qualified=None):
    s = operators()[0]
    relkind = (relkind,) if isinstance(relkind, str) else tuple(relkind)
    c, names = composed_regex(operator, names, not_=not_, case=case)
    execV = [relkind, names]
    if schema:
        execV.append((schema,) if isinstance(schema, str) else tuple(schema))
        a = s('AND n.nspname IN %s')
    else:
        a = s('')

    cursor = conn.cursor()
    cursor.execute(s('SELECT {} FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE relkind IN %s AND relname {} %s {};') \
        .format(composed_parse({'nspname': qualified, 'relname': True}), c, a), execV)
    if qualified:
        return cursor.fetchall()
    else:
        return [x for x, in cursor.fetchall()]


def exmog(cursor, input):
    print(cursor.mogrify(*input))
    cursor.execute(*input)

    return cursor


def composed_regex(operator, names, not_, case):
    s = operators()[0]
    if operator.lower() == 'like':
        c = s('LIKE') if case else s('ILIKE')
        c = s('NOT ')+c+s(' ALL') if not_ else c+s(' ANY')
        if isinstance(names, str):
            names = [names]
        names = (names,)
    elif operator.lower() == 'similar':
        c = s('NOT SIMILAR TO') if not_ else s('SIMILAR TO')
        if not isinstance(names, str):
            names = '|'.join(names)
    elif operator.lower() == 'posix':
        c = s('~')
        if not case:
            c += s('*')
        if not_:
            c = s('!') + c
        if not isinstance(names, str):
            names = '|'.join(names)

    return c, names


def table_exists(conn, name):
    exists = False
    try:
        cur = conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='" + name + "')")
        exists = cur.fetchone()[0]
        cur.close()
    except psg.Error as e:
        print(e)

    return exists


def get_table_colNames(conn, name):
    colNames = []
    try:
        cur = conn.cursor()
        cur.execute("select * from " + name + " LIMIT 0")
        for desc in cur.description:
            colNames.append(desc[0])
        cur.close()
    except psg.Error as e:
        print(e)

    return colNames


def set_comment(conn, tbl, comment, schema=None):
    s, n, _ = operators()
    schema = composed_dot(schema)

    return pg_execute(conn, s('COMMENT ON TABLE {}{} IS %s').format(schema, n(tbl)), values=[str(comment)])


def operators():
    return sql.SQL, sql.Identifier, sql.Placeholder()
