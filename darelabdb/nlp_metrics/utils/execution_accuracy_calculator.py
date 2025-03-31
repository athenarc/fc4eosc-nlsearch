import itertools
import random
import time
from collections import Counter
from typing import Literal, Union

import pandas as pd
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.results_cache.query_results_cache import (
    cache_query_results,
)
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.utils_query_analyzer.process_sql import exists_order_by
from loguru import logger


@cache_query_results
def exponential_backoff_query_execution(sql: str, db: Database) -> pd.DataFrame:
    """
    Executes the given query in the given db with exponential backoff in case the database is in recovery mode.
    The waiting intervals are: 16, 64, 180, 600 seconds

    Args:
        sql (str): The query to execute.
        db (Database): The database upon which the query will be executed.

    Returns:
        The Dataframe with the results and the execution time of the query.
    """
    wait_intervals = [16, 64, 180, 600]
    for wait_interval in wait_intervals:
        result = db.execute(sql=sql, limit=-1)
        if isinstance(result, pd.DataFrame):
            return result
        elif (
            "error" in result
            and "the database system is in recovery mode" in result["error"]
        ):
            logger.warning(
                f"The database is in recovery mode. Waiting {wait_interval} seconds before trying again."
            )
            time.sleep(wait_interval)
        else:
            return result


def _get_results(
    sql: str, db: Database, query_type: Literal["reference", "prediction"]
) -> (pd.DataFrame, float):
    """
    Executes the given query in the given db.

    Args:
        sql (str): The query to execute.
        db (Database): The database upon which the query will be executed.
        query_type (str): The type of query to execute. THe possible options are: 'reference', 'prediction'.
            This parameter is used for the messages in case of an error.

    Returns:
        The Dataframe with the results and the execution time of the query.
    """
    start_time = time.time()

    result = exponential_backoff_query_execution(sql=sql, db=db)
    exec_time = time.time() - start_time

    if "error" in result:
        logger.warning(
            f"There was an error while executing the {query_type} query {sql}! The execution accuracy will be set "
            f"to 0."
        )
        raise SyntaxError(result["error"])

    return result, exec_time


def _constrain_mappings(
    mappings: dict, row: list, results: Union[pd.DataFrame, list]
) -> None:
    """
    Reduce the given mappings based on the constraints in order the given row to be equal with a result row.
    Args:
        mappings: A dictionary with the keys the row columns' indexes and values the results columns' indexes.
        row: A list of column values
        results: A dataframe or a list with which the row is compared.
    """
    row_column_num = len(row)

    if isinstance(results, pd.DataFrame):
        for i in range(row_column_num):
            i_mappings = mappings[i].copy()
            for j in i_mappings:
                if row[i] not in results[j].values:
                    mappings[i].remove(j)
    else: # results is a list (order_matters=True case)
        results_len = len(results)
        for i in range(row_column_num):
            i_mappings = mappings[i].copy()
            for j in i_mappings:
                # Check if j is a valid index for results
                if j >= results_len:
                     mappings[i].remove(j)
                     continue

                # REMOVED STRICT TYPE CHECK - rely on value comparison
                # Add try-except for comparison issues if necessary
                try:
                    if row[i] != results[j]:
                         mappings[i].remove(j)
                except TypeError: # Handle potential comparison errors between types
                     mappings[i].remove(j)


def _valid_final_mapping(possible_mappings: dict) -> bool:
    """
    Checks if a mapping dictionary is valid. A mapping is considered valid if all keys are mapped to 1 or 0 values and
    no value exists more than one times.
    """
    if not all([len(possible_mappings[i]) <= 1 for i in range(len(possible_mappings))]):
        return False

    # Check that there are no duplicates except for None
    values_count = Counter(
        list([value[0] if len(value) else None for value in possible_mappings.values()])
    )
    if None in values_count.keys():
        values_count.pop(None)

    if any(value_count > 1 for value_count in values_count.values()):
        return False

    return True


def _constrain_columns_mappings(
    result1: pd.DataFrame, result2: pd.DataFrame, order_matters: bool
) -> dict:
    """
    Creates a dictionary with the mapping of columns from result1 with result2.
    """

    # Create all possible mappings
    possible_mappings = {
        i: [j for j in range(result2.shape[1])] for i in range(result1.shape[1])
    }
    iterations = 0
    while not _valid_final_mapping(possible_mappings) and iterations < 10:
        stable_row_idx = random.randint(0, result1.shape[0] - 1)

        if order_matters:
            _constrain_mappings(
                possible_mappings,
                result1.iloc[stable_row_idx].tolist(),
                result2.iloc[stable_row_idx].tolist(),
            )

        else:
            _constrain_mappings(
                possible_mappings, result1.iloc[stable_row_idx].tolist(), result2
            )

        iterations += 1
    return possible_mappings


def _map_results(
    result1: pd.DataFrame, result2: pd.DataFrame, mappings: dict
) -> (pd.DataFrame, pd.DataFrame):
    # Remove columns that are not mapped
    mappings = {k: v for k, v in mappings.items() if v is not None}

    result1 = result1[list(mappings.keys())]
    result1 = result1.rename(columns=mappings)

    if len(result2.columns) != len(result1.columns):
        result2 = result2[list(mappings.values())]

    return result1, result2


def _valid_column_mapping(comb: dict) -> bool:
    # If the combination contains only one column and it is None or the combination contain only 1 value
    if (len(comb) == 1 and list(comb.values())[0] is None) or set(comb.values()) == {
        None
    }:
        return False

    # Check that each column is used at most once in the mapping values
    value_count = Counter(list(comb.values()))
    value_count.pop(None, None)

    if sum(value_count.values()) != len(value_count.values()):
        return False

    return True


def _get_valid_column_mappings(column_possible_mappings: dict) -> list[dict]:
    """Returns the column mapping combinations based on which 2 dataframes can be compared"""

    # Add None as possible mapping value to all values
    column_possible_mappings = {
        k: v + [None] for k, v in column_possible_mappings.items()
    }

    # Get all permutations
    columns_combinations = (
        list(itertools.product(*column_possible_mappings.values()))
        if len(column_possible_mappings) > 1
        else list(column_possible_mappings.values())
    )
    column_mapping_combs = [
        {
            k: v
            for k, v in zip(
                list(column_possible_mappings.keys()), list(columns_combination)
            )
        }
        for columns_combination in columns_combinations
    ]

    # Remove invalid combinations
    column_mapping_combs = [
        column_valid_comb
        for column_valid_comb in column_mapping_combs
        if _valid_column_mapping(column_valid_comb)
    ]

    # Order combinations with descending number of None values
    sorted_column_mapping_combs = sorted(
        column_mapping_combs, key=lambda x: Counter(x.values())[None]
    )

    return sorted_column_mapping_combs

def _dataframes_approx_equal(df1: pd.DataFrame, df2: pd.DataFrame, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    """Check if two DataFrames are approximately equal within given tolerances."""
    try:
        pd.testing.assert_frame_equal(
            df1, df2,
            check_exact=False,
            atol=atol,
            rtol=rtol,
            check_dtype=False,
            check_column_type=False
        )
        return True
    except AssertionError:
        return False
    
def _results_comparison(
    result1: pd.DataFrame, result2: pd.DataFrame, order_matters: bool,atol: float = 1e-5, rtol: float = 1e-5
) -> Literal[
    "equal", "columns_subset", "columns_superset", "columns_intersect", "different"
]:
    if result1.shape[0] == 0 and result2.shape[0] == 0:
        return "equal"

    if result1.shape[0] != result2.shape[0]:
        return "different"

    # Remove column names
    result1.columns = [i for i in range(result1.shape[1])]
    result2.columns = [i for i in range(result2.shape[1])]

    # Compare results
    if _dataframes_approx_equal(result1, result2, atol, rtol):
        return "equal"

    columns_num_difference = len(result1.columns) - len(result2.columns)

    if columns_num_difference > 0:
        columns_mappings = _constrain_columns_mappings(result1, result2, order_matters)
        reverse = False
    else:
        columns_mappings = _constrain_columns_mappings(result2, result1, order_matters)
        reverse = True

    for column_mapping_comb in _get_valid_column_mappings(columns_mappings):
        # Map results
        if not reverse:
            result1_edited, result2_edited = _map_results(
                result1, result2, column_mapping_comb
            )
        else:
            result2_edited, result1_edited = _map_results(
                result2, result1, column_mapping_comb
            )

        # Order the results to compare them
        if not order_matters:
            columns = list(result1_edited.columns)
            result1_edited = result1_edited.sort_values(by=columns).reset_index(
                drop=True
            )
            result2_edited = result2_edited.sort_values(by=columns).reset_index(
                drop=True
            )

        none_mappings_num = list(column_mapping_comb.values()).count(None)
        if _dataframes_approx_equal(result1_edited, result2_edited, atol, rtol):
            if none_mappings_num > 0:
                if columns_num_difference != 0:
                    return "columns_superset" if not reverse else "columns_subset"
                else:
                    return "columns_intersect"
            else:
                return "equal"

    return "different"


def exec_evaluator(db_name: str, pred: str, target: str) -> dict:
    """
    Returns the execution accuracy result for the given prediction and target sql queries.
    Args:
        db_name: The name of the database upon which the queries will run. It can be one of the 3 cases below:
            * Path of a sqlite database ending in .sqlite or .db
            * A hosted database such as "fc4eosc". For available databases check the utils_configs component
            * A hosted database with a specified schema such as "fc4eosc.fc4eosc_subset"
        pred: The predicted sql query.
        target: The target sql query.
    Returns: A dictionary with the execution accuracy results. (e.g., {"exec": 0, "exec-only_common_result_columns": 1})
    """

    # Connect to database
    if db_name.endswith(".sqlite") or db_name.endswith(".db"):
        # Path of a sqlite database
        db = DatabaseSqlite(db_name)
        db_dialect = "sqlite"
    elif "." in db_name:
        # Hosted database with a specified schema
        db_name, schema = db_name.split(".")
        db = Database(db_name, specific_schema=schema)
        db_dialect = "postgres"
    else:
        # Hosted database, no schema specified
        db = Database(db_name)
        db_dialect = "postgres"

    # Get the results of the predicted query
    pred_results, pred_exec_time = _get_results(
        sql=pred, db=db, query_type="prediction"
    )

    # Get the results of the target query
    target_results, target_exec_time = _get_results(
        sql=target, db=db, query_type="reference"
    )

    # Check if the order of the results matter
    order_matters = exists_order_by(target, db_dialect)

    # Compare the results
    comp = _results_comparison(pred_results, target_results, order_matters)

    exec_results = {
        "exec": 1 if comp == "equal" else 0,
        "exec-only_common_result_columns": 0 if comp == "different" else 1,
        "exec-target_result_columns_subset": (
            1 if comp in ["equal", "columns_subset"] else 0
        ),
        "exec-target_result_columns_superset": (
            1 if comp in ["equal", "columns_superset"] else 0
        ),
        "target_exec_time": target_exec_time,
        "pred_exec_time": pred_exec_time,
    }

    return exec_results
