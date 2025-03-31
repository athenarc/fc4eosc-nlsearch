import re

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from sql_metadata import Parser


def sql_remove_table_aliases(datapoint: SqlQueryDatapoint) -> SqlQueryDatapoint:
    """
    Replaces the alias of a datapoint's SQL query with the actual table names.
    Args:
        datapoint (SqlQueryDatapoint): The datapoint that contains a sql query in which we apply the processing.

    Returns (SqlQueryDatapoint): The datapoint with the updated sql query.
    """

    sql_query = datapoint.sql_query
    parser = Parser(sql_query)

    # For every table with an alias
    for table_alias, table_original in parser.tables_aliases.items():
        # Remove the alias declaration using regexp to guarantee that the case
        # of the AS statement is ignored
        compiled = re.compile(re.escape(f" AS {table_alias}"), re.IGNORECASE)
        sql_query = compiled.sub("", sql_query)

        # Replace all uses of the alias with the original name
        sql_query = sql_query.replace(table_alias, table_original)

    datapoint.sql_query = sql_query
    return datapoint
