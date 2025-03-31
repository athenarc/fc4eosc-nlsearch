from typing import Optional
from sqlglot import transpile
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def sql_transpile(
    datapoint: SqlQueryDatapoint,
    read_dialect: str,
    write_dialect: str,
    field_name: Optional[str] = None,
    expect_candidates: bool = False,
) -> SqlQueryDatapoint:
    """
    Transpiles SQL queries from one dialect to another (e.g., MySQL to Postgres)
    This is handled by the `sqlglot` library, a list of supported dialects can
    be found in its [documentation](https://sqlglot.com/sqlglot/dialects/dialect.html#Dialects).

    Args:
        datapoint (SqlQueryDatapoint): A datapoint to process.
        read_dialect (str): The dialect of the given SQL query.
        write_dialect (str): The dialect to which the query will be transpiled.
        field_name (Optional[str]): The field in the datapoint that contains the
            SQL query or SQL query candidates. If not given, it will be set to
            `prediction` if `expect_candidates` is False or `candidates` if
            `expect_candidates` is True. Defaults to None.
        expect_candidates (bool, optional): Whether the step should expect multiple
            candidates instead of a single SQL. Defaults to False.

    Returns:
        SqlQueryDatapoint: _description_
    """
    if field_name is None:
        # If no field_name is specified, set it based on expect_candidates
        if expect_candidates:
            field_name = "candidates"
        else:
            field_name = "prediction"

    if expect_candidates:
        sql_candidates = getattr(datapoint, field_name)
        new_candidates = [
            transpile(sql_query, read=read_dialect, write=write_dialect)[0]
            # sqlglot.transpile returns a list where each element is a SQL statement
            for sql_query in sql_candidates
        ]
        setattr(datapoint, field_name, new_candidates)
    else:
        sql_query = getattr(datapoint, field_name)
        new_sql_query = transpile(sql_query, read=read_dialect, write=write_dialect)
        # sqlglot.transpile returns a list where each element is a SQL statement
        new_sql_query = new_sql_query[0]
        setattr(datapoint, field_name, new_sql_query)

    return datapoint
