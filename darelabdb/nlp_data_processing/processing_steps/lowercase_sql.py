import sqlparse
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def lowercase_sql(datapoint: SqlQueryDatapoint) -> SqlQueryDatapoint:
    """
    Lowercase the given datapoint's sql query.

    Args:
        datapoint (SqlQueryDatapoint): The datapoint that contains a sql query in which we apply the processing.

    Returns:
        SqlQueryDatapoint: The datapoint with the updated sql query.
    """
    # Lower-case non value tokens & reindex
    datapoint.sql_query = sqlparse.format(
        datapoint.sql_query, keyword_case="lower", identifier_case="lower"
    )
    return datapoint
