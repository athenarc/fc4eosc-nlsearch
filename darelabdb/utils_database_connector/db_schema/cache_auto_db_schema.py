import json
import os
from pathlib import Path

from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.db_schema.auto_db_schema import (
    obtain_schema_from_db,
)
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite

SCHEMA_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/schema_cache"


def schema_exists(schema_name: str) -> bool:
    if os.path.exists(f"{SCHEMA_CACHE_PATH}/schemas.json"):
        with open(f"{SCHEMA_CACHE_PATH}/schemas.json", "r") as f:
            schemas = json.load(f)
            return schema_name in schemas


def get_schema(
    database_str: str,
    sample_size: int = 20,
    infer_foreign_keys: bool = False,
    enable_cache: bool = True,
) -> list:
    """
    Provides the schema of a database given the database string. The database string can an existing database
    i.e. "fc4eosc" or a sqlite database path ending with .db or .sqlite

    Args:
        database_str: The database string
        sample_size: The sample size to return if the column is not categorical
        infer_foreign_keys: Whether to infer foreign keys or not based on the column names
        enable_cache: A boolean to enable caching of the schema. If disable no access of the cache will be performed.

    Returns:
        list: The schema of the database in the following format
            ```
            [
                {
                    "table_name": "table1",
                    "columns": [
                        {
                            "column": "col1",
                            "values": [1, 2, 3, 4, 5],
                            "data_type": "INTEGER",
                            "is_pk": False,
                            "foreign_keys": [
                                {
                                    "foreign_table": "table2",
                                    "foreign_column": col3,
                                },
                                ...
                            ],
                        },
                        ...
                    ]
                },
                ...
            ]
            ```
    """
    # We want to separate schemas with different number of example values
    schema_id = f"{database_str}_{str(sample_size)}_{str(infer_foreign_keys)}"

    if schema_exists(schema_id) and enable_cache:
        with open(f"{SCHEMA_CACHE_PATH}/schemas.json", "r") as f:
            return json.load(f)[schema_id]

    if database_str.endswith(".db") or database_str.endswith(".sqlite"):
        db = DatabaseSqlite(database_str)
    elif "." in database_str:
        db_str, schema = database_str.split(".")
        db = Database(database=db_str, specific_schema=schema)
    else:
        db = Database(database=database_str)

    schema = obtain_schema_from_db(db, sample_size, infer_foreign_keys)

    if enable_cache:
        store_schema(schema, schema_id)

    return schema


def store_schema(schema: list, schema_id: str) -> None:
    if os.path.exists(f"{SCHEMA_CACHE_PATH}/schemas.json"):
        with open(f"{SCHEMA_CACHE_PATH}/schemas.json", "r") as f:
            schemas = json.load(f)
    else:
        if not os.path.exists(SCHEMA_CACHE_PATH):
            os.makedirs(SCHEMA_CACHE_PATH)
        schemas = {}

    schemas[schema_id] = schema

    with open(f"{SCHEMA_CACHE_PATH}/schemas.json", "w") as f:
        json.dump(schemas, f, default=str, indent=2)


if __name__ == "__main__":
    # get_schema("Car_Database.db", infer_foreign_keys=True)
    get_schema("fc4eosc", sample_size=20, infer_foreign_keys=True)
