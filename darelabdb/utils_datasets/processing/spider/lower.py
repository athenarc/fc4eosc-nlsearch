import sqlparse


def lowercase_query(sql_query: str) -> str:
    """
    Turn the given sql_query into lowercase, without altering any values that
    appear in the query.

    Args:
        sql_query (str): A string of the SQL Query to be turned into lowercase.

    Returns:
        str: The SQL query in lowercase, except for the values.
    """

    sql_query = sqlparse.format(
        sql_query, keyword_case="lower", identifier_case="lower"
    )

    return sql_query
