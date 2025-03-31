import pandas as pd
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.db_schema.auto_db_schema import (
    obtain_schema_from_db,
)
from darelabdb.utils_database_connector.db_schema.filter_schema import filter_schema
from darelabdb.utils_query_analyzer.process_sql import (
    add_column_in_select,
    base_table_exists,
    get_base_query_table_aliases,
    get_base_tables,
    get_select_attributes,
    remove_select_column_aliases,
)
from loguru import logger
from sqlglot import parse_one


def get_filtered_fc4e_schema(excluded_columns):
    # Get the database schema
    fc4e_db_schema = obtain_schema_from_db(
        Database("fc4eosc", specific_schema="fc4eosc_subset"), sample_size=10
    )
    # Filter out columns that are not needed
    fc4e_db_schema = filter_schema(fc4e_db_schema, exclude_values=excluded_columns)

    return fc4e_db_schema


def remove_sk_ids_from_results(results: pd.DataFrame) -> pd.DataFrame:
    kept_columns = [col for col in results.columns if "sk_id" not in col]
    return results[kept_columns]


def inject_id_in_select_if_title_exists(query: str) -> str:
    if not base_table_exists(query, "result"):
        return query

    tables = get_base_tables(parse_one(query))

    table_aliases = get_base_query_table_aliases(query)
    result_alias = ""
    for alias, base_table in table_aliases.items():
        if base_table == "result":
            result_alias = alias + "."
            break

    select_attributes = get_select_attributes(query)

    # Check that the query has a title column
    if not any("title" in col for col in select_attributes):
        return query

    # Check that the query does not have an id column of the result table yet
    if (
        any(
            (result_alias + "id" == col or "result.id" == col)
            for col in select_attributes
        )
        or "id" in select_attributes
    ):
        return query

    # Check if there is a prefix in the title column
    prefix = ""
    for col in select_attributes:
        if ".title" in col:
            prefix = col.split(".")[0] + "."
            break

    query = add_column_in_select(query, f"{prefix}id")

    return query


def executeSQL(query, database, limit=10):
    query = inject_id_in_select_if_title_exists(query)

    db = Database(database)
    options = {"database": database, "query": query, "limit": limit}

    try:
        logger.info(f"Executing query: {query}")
        resp = db.execute(query, limit=limit)
        if isinstance(resp, dict) and "error" in resp:
            return resp["error"]

        resp = remove_sk_ids_from_results(resp)
        return resp

    # TODO: Better exception handling
    except Exception as e:
        raise Exception(f"Query Execution error for {options}")
