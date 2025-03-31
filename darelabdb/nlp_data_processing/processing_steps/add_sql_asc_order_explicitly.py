import re

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def add_sql_asc_order_explicitly(datapoint: SqlQueryDatapoint) -> SqlQueryDatapoint:
    """
    Adds to the given datapoint's sql query the keyword asc in all columns of order by with
    no explicit order.

    Args:
        datapoint (SqlQueryDatapoint): The datapoint that contains a sql query in which we apply the processing.

    Returns:
        SqlQueryDatapoint: The datapoint with the updated sql query.
    """
    # Get the content of all the order by clauses
    order_by_pattern = re.compile(
        r"\bORDER\s+BY\b\s+(.*?)(?=\b(?:LIMIT|OFFSET|;|$|\)))",
        re.IGNORECASE | re.DOTALL,
    )

    modified_sql_query = datapoint.sql_query
    for orderby_clause_content in order_by_pattern.finditer(datapoint.sql_query):
        # Get all the columns
        columns = orderby_clause_content.group(1).split(",")

        modified_order_by = []
        for column in columns:
            # Check if the column has an order
            if len(column.split()) == 1:
                modified_order_by.append(f"{column} asc")
            else:
                modified_order_by.append(column)

        modified_order_by = ",".join(modified_order_by)

        # Update the content of the order by clause
        modified_sql_query = modified_sql_query.replace(
            orderby_clause_content.group(), f"order by {modified_order_by}"
        )

    datapoint.sql_query = modified_sql_query
    return datapoint
