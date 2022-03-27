DO $$
DECLARE
   counter    INTEGER := 1;
   first_name VARCHAR(50) := 'John';
   last_name  VARCHAR(50) := 'Doe';
   payment    NUMERIC(11,2) := 20.5;
BEGIN
   RAISE NOTICE '% % % has been paid % USD', counter, first_name, last_name, payment;
END $$;

[x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])]


sorted(range(len(s)), key=lambda k: s[k])

numpy.argsort()

from operator import itemgetter
indices, L_sorted = zip(*sorted(enumerate(L), key=itemgetter(1)))


{k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

df1 = df1.assign(e=pd.Series(np.random.randn(sLength)).values)


CREATE OR REPLACE FUNCTION sig_digits(n anyelement, digits int)
RETURNS numeric
AS $$
    SELECT round(n, digits - 1 - floor(log(abs(n)))::int)
$$ LANGUAGE sql IMMUTABLE STRICT;


pip list --outdated --format=freeze | %{$_.split('==')[0]} | %{pip install --upgrade $_}

/*pga4dash*/
SELECT
    pid,
    datname,
    usename,
    application_name,
    client_addr,
    pg_catalog.to_char(backend_start, 'YYYY-MM-DD HH24:MI:SS TZ') AS backend_start,
    state,
    wait_event_type || ': ' || wait_event AS wait_event,
    pg_catalog.pg_blocking_pids(pid) AS blocking_pids,
    query,
    pg_catalog.to_char(state_change, 'YYYY-MM-DD HH24:MI:SS TZ') AS state_change,
    pg_catalog.to_char(query_start, 'YYYY-MM-DD HH24:MI:SS TZ') AS query_start,
    backend_type,
    CASE WHEN state = 'active' THEN ROUND((extract(epoch from now() - query_start) / 60)::numeric, 2) ELSE 0 END AS active_since
FROM
    pg_catalog.pg_stat_activity
WHERE
    datname = (SELECT datname FROM pg_catalog.pg_database WHERE oid = 16394)ORDER BY pid
