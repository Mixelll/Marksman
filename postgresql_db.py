"""
This module is intended for the creation and execution of `psycopg2`
composables, representing queries-pending-values. The composables are further
composed together to produce PostgreSQL queries. It leverages nesting, parsing
and keywords to generate complex queries with properly escaped names.

Parameters are denoted by :param_or_type:, brackets in conjunction with this
notation have the same meaning as when used as a Python literal.

The list of major parameters recurring in the module:

:param columns:
    :[str or [str]]: List of strings[] representing column names
    passed to `composed_columns`.

:param schema:
    :str: representing schema name

:param returning:
    :[str or NestedIterable]: parsable by `composed_parse` representing
    returned expressions.

:param conflict:
    :[str]: representing column names following SQL's `ON CONFLICT`

:param nothing:
    Wether to `DO NOTHING` following SQL's `ON CONFLICT`

:param set:
    :[str: or Iterable]:, parsables passed to `composed_set`

:param parse:
    :Callable: passed to `composed_set` and `composed_separated`
    as `parse=parse`, usually `composed_parse` is used.

:param where:
    :[str or NestedIterable]:, parsable passed to `composed_parse`
"""






# from inspect import getmembers, isfunction, isclass
import io
import pandas as pd
import psycopg2 as psg

import marksman_extras as me

from psycopg2 import sql
from sqlalchemy import create_engine
from uuid import uuid4


# execute connection objects
with open('connections.txt') as f:
    exec(f.read())
conn
engine

def psg_operators():
    return sql.SQL, sql.Identifier, sql.Placeholder()

approvedExp = ['primary key', 'foreign key', 'references', 'default', 'uuid',
                'on delete restrict', 'unique', 'timestamp with time zone',
                'on delete cascade', 'current_timestamp','double precision',
                'not null', 'json', 'bytea', 'timestamp', 'like', 'and',
                'or', 'join', 'inner', 'left', 'right', 'full', '', '.', ',',
                 '=', '||','++','||+', 'min', 'max', 'least', 'greatest', 'like']

def get_table_as_df(connOrEngine, tbl, columns=None, schema=None, **kwargs):
    return query_get_df(connOrEngine, sql.SQL('SELECT {} FROM {}{}') \
        .format(composed_columns(columns), composed_dot(schema),
                sql.Identifier(tbl)), **kwargs)


def query_get_df(connOrEngine, query):
    try:
        return pd.read_sql_query(query, con=connOrEngine)
    except:
        cur = connOrEngine.cursor()
        cur.execute(query)
        return pd.DataFrame(cur.fetchall())


def pg_execute(conn, query, values=None, commit=True, as_string=False, mogrify=False):
    """
    Execute a psycopg2 composable, possibly containing a placeholder -
    *sql.Placeholder* or `'%s'` for the `values`.

    :param as_string:
        Wether to print the output of `conn.cursor.as_string(...)`

    :param mogrify:
        Wether to print the output of `conn.cursor.mogrify(...)`

    :return:
        :[(...)]:, returns array of tuples (rows)
    """
    cur = conn.cursor()
    if as_string:
        print(query.as_string(conn))
    if mogrify:
        print(cur.mogrify(query, values))
    cur.execute(query, values)
    if commit:
        conn.commit()
    if cur.description is not None:
        return cur.fetchall()


def composed_columns(columns, enclose=False, parse=None, literal=None, **kwargs):
    s = psg_operators()[0]
    if parse is None:
        parse = lambda x: composed_separated(x, '.', **kwargs)
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        return s('*')
    else:
        comp = s(', ').join(map(parse, columns))
        if enclose:
            return s('({})').format(comp)
        return comp


def composed_parse(exp, safe=False, tuple_parse=composed_columns):
    """
    Parse a nested container of strings by recursively pruning according
    to the following rules:

    - Enclose expression with `'$'` to parse the string raw into the quary,
        only selected expressions are allowed. *
    - If exp is  `'%s'` or *sql.Placeholder* to parse into *sql.Placeholder*.
    - If exp is a tuple it will be parsed by `composed_columns`.
    - If exp is a dict the keys will be parsed by `composed_columns` only if
        exp[key] evaluates to True.
    - Else (expecting an iterable) `composed_parse` will be applied to
        each element in the iterable.

    :param safe:
        Wether to disable the raw parsing in *

    :param exp:
        :[:str: or :Iterable:] or str: List (or string) of parsables
        expressions (strings or iterables).

    :param enclose:
        :Bool: passed to `composed_columns` as `enclose=enclose`

    :param parse:
        :Callable: passed to `composed_columns` as `parse=parse`,
        usually `composed_parse` itself is used.

    :return:
        :Composable:
    """
    s, n, p = psg_operators()

    if isinstance(exp, str):
        if not safe and exp[0]=='$' and exp[-1]=='$':
            exp = exp.replace('$','')
            if exp.strip(' []()').lower() in approvedExp:
                returned =  s(exp)
            elif exp.strip(' []()') == '%s':
                e = exp.split('%s')
                returned = s(e[0]) + p + s(e[1])
            else:
                raise ValueError(f'Expression: {exp.strip(" []()")} not found \
                                in allowed expressions')
        elif exp.strip('$ []()') == '%s':
            e = exp.replace('$','').split('%s')
            returned = s(e[0]) + p + s(e[1])
        else:
            returned =  n(exp)
    elif isinstance(exp, sql.Placeholder):
        returned = exp
    elif isinstance(exp, tuple):
        returned = tuple_parse(filter(me.mbool, exp))
    elif isinstance(exp, dict):
        returned = tuple_parse(filter(me.mbool, [k for k in exp.keys() if exp[k]]))
    else:
        expPrev = exp[0]
        for x in exp[1:]:
            if x == expPrev:
                raise ValueError(f"Something's funny going on - {x,x} pattern is repeated ")
            else:
                expPrev = x

        return sql.Composed([composed_parse(x, safe=safe, tuple_parse=tuple_parse) for x in filter(me.mbool, exp)])

    return s(' {} ').format(returned)


def composed_insert(tbl, columns, schema=None, returning=None, conflict=None,
                    nothing=False, set=None, parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"schema"."tbl"`

    :return:
        :Composable:, query awaiting values
    """
    s, n, p = psg_operators()

    comp = s('INSERT INTO {}{} ({}) VALUES {} ').format(composed_dot(schema),
            n(tbl), composed_separated(columns), p)

    if conflict:
        if nothing:
            comp += s('ON CONFLICT ({}) DO NOTHING').format(n(conflict))
        else:
            if set is None:
                set = columns
            comp += s('ON CONFLICT ({}) DO UPDATE').format(n(conflict)) + composed_set(set, parse=parse)

    if returning is not None:
        comp += s(' RETURNING {}').format(composed_separated(returning, parse=parse))

    return comp


def composed_update(tbl, columns, returning=None, schema=None, where=None,
                    parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"schema"."tbl"`

    :return:
        :Composable:, query awaiting values
    """
    s, n, p = psg_operators()

    comp = s('UPDATE {}').format(composed_dot(schema))
    comp += n(tbl) + composed_set(columns)

    if where is not None:
        comp += s(' WHERE {}').format(parse(where))

    if returning is not None:
        comp += s(' RETURNING {}').format(parse(returning))

    return comp


def composed_create(tbl, columns, schema=None, like=None,
                    inherits=None, constraint=None, parse=composed_parse):
    """
    Create a table as `"schema"."tbl"`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param constraint:
        :[str or NestedIterable]:, table constraints
        parsable by `composed_parse`, passed to `composed_columns`

    :return:
        :Composable:, full create table query
    """
    s, n, p = psg_operators()

    if isinstance(columns[0], str):
        columns = [columns]
    comp = s('CREATE TABLE {}{} (').format(composed_dot(schema), n(tbl))

    if like is not None:
        comp += composed_columns(like, parse=lambda x:
            s('LIKE {} INCLUDING ALL, ').format(composed_separated(x, '.', )))

    if constraint:
        if isinstance(constraint[0], str): constraint = [constraint]
        comp += composed_columns(constraint, parse=lambda x:
            s('CONSTRAINT ({}), ').format(parse(x)))

    comp += composed_columns(columns, parse=parse) + s(') ')

    if inherits is not None:
        comp += s('INHERITS ({})').format(composed_columns(inherits))

    return comp


def composed_select_from_table(tbl, columns=None, schema=None):
    """
    Select columns from table as `"schema"."tbl"`

    :return:
        :Composable:, full select query which can be further used to compose
    """
    return sql.SQL('SELECT {} FROM {}{} ').format(composed_columns(columns),
                    composed_dot(schema), sql.Identifier(tbl))


def composed_from_join(join=None, tables=None, columns=None, using=None, parse=composed_parse):
    s, _, _ = psg_operators()
    def n(x): composed_separated(x, '.')

    joinc = []
    for v in multiply_iter(join, max(iter_length(tables, columns, using))):
        vj = '$'+v+'$' if v else v
        joinc.append(parse([vj, '$ JOIN $']))

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
            comp += s('NATURAL ') + parse([join, '$ JOIN $']) + \
                    s('{} ').format(n(table[i]))
    elif columns:
        columns = list(columns)
        comp = s('FROM {} ').format(n(columns[0][:-1]))
        for i in range(1, len(columns)):
            toMap = columns[i][:-1], columns[i-1], columns[i-1]
            comp += joinc[i-1] + s('{} ON {} = {} ').format(*map(n, toMap))
    else:
        raise ValueError("Either tables or columns need to be given")

    return comp


def composed_set(set, parse=composed_parse):
    """
    Return a composable of the form `SET (...) = (...)`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param set:
        :[str or NestedIterable]:, set table columns
        parsable by `composed_parse`, passed to `composed_columns` and
         `composed_separated`

    :return:
        :Composable:
    """
    s, n, p = psg_operators()
    if not set:
        return s('')
    col, val = [], []
    for c in set:
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
    return formatted.format(composed_columns(col),
                            composed_separated(val, parse=parse))


def composed_between(start=None, end=None):
    """
    Return a composable that compares values to `start` and `end`

    :param start:
        :str or datetime or numeric:

    :param end:
        :str or datetime or numeric:

    :return:
        :(Composable, Array):, composable and values passed to `pg_execute` are returned
    """
    s = psg_operators()[0]
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
    s, n, _ = psg_operators()
    if name:
        if not isinstance(name, str):
            return [composed_dot(x) for x in name]
        return s('{}.').format(n(name))
    return s('')



def composed_separated(names, sep=', ', enclose=False, AS=False, parse=None):
    s, n, _ = psg_operators()
    if parse is None:
        parse = n
    if isinstance(names, str):
        names = [names]
    names = list(filter(me.mbool, names))
    if sep in [',', '.', ', ', ' ', '    ']:
        comp = s(sep).join(map(parse, names))
        if AS:
            comp += s(' ') + n(sep.join(names))
        if enclose:
            return s('({})').format(comp)
        return comp
    else:
        raise ValueError(f'Expression: "{sep}" not found in approved separators')


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
    s, n, _ = psg_operators()
    conn = engine.raw_connection()
    df.head(0).to_sql(tbl, engine, if_exists='replace', index=index, schema=schema)
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=index)
    output.seek(0)

    temp, schema, tbl = n(tbl + '_' + str(uuid4())[:8]), composed_dot(schema), n(tbl)
    cur.execute(s('CREATE TEMP TABLE {} (LIKE {}{} INCLUDING ALL);').format(temp, schema, tbl))
    cur.copy_expert(s("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(temp), output)
    cur.execute(s('DELETE FROM {}{} WHERE ({index}) IN (SELECT {index} FROM {});')
        .format(schema, tbl, temp, index=composed_separated(tuple(df.index.names))))
    cur.execute(s('INSERT INTO {}{} SELECT * FROM {};').format(schema, tbl, temp))
    cur.execute(s('DROP TABLE {};').format(temp))
    conn.commit()


def get_tableNames(conn, names, operator='like', not_=False, relkind=('r', 'v'),
                    case=False, schema=None, qualified=None):
    s = psg_operators()[0]
    relkind = (relkind,) if isinstance(relkind, str) else tuple(relkind)
    c, names = composed_regex(operator, names, not_=not_, case=case)
    execV = [relkind, names]
    if schema:
        execV.append((schema,) if isinstance(schema, str) else tuple(schema))
        a = s('AND n.nspname IN %s')
    else:
        a = s('')

    cursor = conn.cursor()
    cursor.execute(s('SELECT {} FROM pg_class c JOIN pg_namespace n ON \
                    n.oid = c.relnamespace WHERE relkind IN %s AND relname {} %s {};') \
        .format(composed_parse({'nspname': qualified, 'relname': True}, safe=True), c, a), execV)
    if qualified:
        return cursor.fetchall()
    else:
        return [x for x, in cursor.fetchall()]


def exmog(cursor, input):
    print(cursor.mogrify(*input))
    cursor.execute(*input)

    return cursor


def composed_regex(operator, names, not_, case):
    s = psg_operators()[0]
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
        cur.execute(f"select exists(select relname from pg_class where relname='{name}')")
        exists = cur.fetchone()[0]
        cur.close()
    except psg.Error as e:
        print(e)

    return exists


def get_table_colNames(conn, name):
    colNames = []
    try:
        cur = conn.cursor()
        cur.execute(f'select * from {name} LIMIT 0')
        for desc in cur.description:
            colNames.append(desc[0])
        cur.close()
    except psg.Error as e:
        print(e)

    return colNames


def set_comment(conn, tbl, comment, schema=None):
    s, n, _ = psg_operators()
    schema = composed_dot(schema)
    query = s('COMMENT ON TABLE {}{} IS %s').format(schema, n(tbl))
    return pg_execute(conn, query, values=[str(comment)])
