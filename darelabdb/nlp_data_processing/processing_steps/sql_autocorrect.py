from typing import Dict
from loguru import logger

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from sqlglot import exp, parse_one
from thefuzz import process


def _correct_attribute_names(
    sql_query: str, db_schema: Dict, table_names_field: str, column_names_field: str
) -> str:
    # NOTE: A case where parsing fails is when the model predicts a column name
    #   with spaces (e.g., `select song name, song release year from singer`)
    try:
        # NOTE: The MySQL dialect is used because Spider doesn't differentiate single/double quotes
        parse = parse_one(sql_query, dialect="mysql")
    except Exception as e:
        logger.warning(f"SQL query '{sql_query}' could not be parsed")
        return None

    # Identify table and column names used in the predicted SQL
    used_tables = [table.name for table in parse.find_all(exp.Table)]
    used_columns = [column.name for column in parse.find_all(exp.Column)]

    # Grab the table and column names that are present in the schema
    table_names = db_schema[table_names_field]
    column_names = [
        column_name for table_idx, column_name in db_schema[column_names_field]
    ]

    # Find the used attributes that do not exist in the schema
    wrong_tables = [table for table in used_tables if table not in table_names]
    wrong_columns = [column for column in used_columns if column not in column_names]

    # For each used attribute that does not exist in the schema, find the most
    # similar that actually exists in the schema and replace it
    corrections = {}
    for wrong_table in wrong_tables:
        predicted_table, score = process.extractOne(wrong_table, table_names)
        # print(f"{wrong_table} -> {predicted_table}")
        corrections[wrong_table] = predicted_table

    for wrong_column in wrong_columns:
        predicted_column, score = process.extractOne(wrong_column, column_names)
        # print(f"{wrong_column} -> {predicted_column}")
        corrections[wrong_column] = predicted_column

    # NOTE: Maybe we should consider the table that the column belongs to in the
    #   prediction to adjust the scores
    # NOTE: The score given by the fuzzy match can be very useful
    # NOTE: Maybe corrections should be made while traversing the SQL for better
    #   results

    # Replace wrong tables and columns with the predicted corrections
    corrected_sql_query = sql_query
    for wrong_name, corrected_name in corrections.items():
        corrected_sql_query = corrected_sql_query.replace(wrong_name, corrected_name)

    return corrected_sql_query


def correct_attribute_names(
    datapoint: SqlQueryDatapoint,
    column_names_field: str = "column_names",
    table_names_field: str = "table_names",
    expect_candidates: bool = False,
) -> SqlQueryDatapoint:
    """
    Automatically correct attribute names that do not exist in the DB schema.

    Args:
        datapoint (SqlQueryDatapoint): A datapoint with an SQL query prediction.
        column_names_field (str, optional): The field that contains the column
            names in the DB schema dict. Defaults to "column_names".
        table_names_field (str, optional): The field that contains the table
            names in the DB schema dict. Defaults to "table_names".
        expect_candidates (bool, optional): Whether to expect multiple candidates,
            instead of a single prediction. Deafults to False.

    Returns:
        SqlQueryDatapoint: The updated datapoint with the corrected SQL prediction.
    """

    if expect_candidates:
        corrected_candidates = [
            _correct_attribute_names(
                candidate,
                datapoint.db_schema,
                table_names_field,
                column_names_field,
            )
            for candidate in datapoint.candidates
        ]
        datapoint.candidates = corrected_candidates
    else:
        corrected_sql_query = _correct_attribute_names(
            datapoint.prediction,
            datapoint.db_schema,
            table_names_field,
            column_names_field,
        )
        datapoint.prediction = corrected_sql_query

    return datapoint
