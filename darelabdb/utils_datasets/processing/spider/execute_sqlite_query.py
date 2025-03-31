import sqlite3

import pandas as pd


def execute_sqlite_query(sql_query: str, sqlite_path: str):
    """
    Returns the result after executing the query in the database

    Args:
        sql_query (str): The query to be executed.
        sqlite_path (str): The path of the database to execute the query.

    Returns:
        pd.DataFrame: The execution result of the given sql_query in the database.
    """
    # Connect to the database and create a cursor
    db_conn = sqlite3.connect(sqlite_path)
    cursor = db_conn.cursor()

    # Execute the query and fetch results
    cursor.execute(sql_query)
    results = pd.DataFrame(cursor.fetchall())

    # Close the cursor and the connection with the database
    cursor.close()
    db_conn.close()

    return results
