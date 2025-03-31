import json
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.utils_database_connector.db_schema import cache_auto_db_schema
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.processing.spider.execute_sqlite_query import (
    execute_sqlite_query,
)
from darelabdb.utils_datasets.processing.spider.lower import lowercase_query
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_files

HUGGINGFACE_REPO = "spider"
BRANCH = "script"


def group_by_sql_query(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and groups rows that have the same SQL query in the field
        `query`. For every other value of the grouped rows, the newly created
        row will have a list of the values from each of the grouped rows, if the
        values are not the same. If the values are the same, then only one
        instance will be kept in the new row

    Args:
        df (pd.DataFrame): A DataFrame on which to group the rows.

    Returns:
        pd.DataFrame: A new DataFrame with grouped rows.
    """
    # Group the DataFrame based on the query and the db_id
    # NOTE: The db_id is needed because it is possible the same SQL exists for multiple DBs
    example_groups = df.groupby(["query", "db_id"])

    data = {column: [] for column in df.columns}

    # Create new DataFrame to return
    new_df = pd.DataFrame(columns=df.columns)

    for idx, group in example_groups:
        for key in data.keys():
            values = group[key].to_list()
            if all(value == values[0] for value in values):
                # If grouped values are the same keep only the first
                data[key].append(values[0])
            else:
                # If grouped values are different keep as a list
                data[key].append(values)

    new_df = pd.DataFrame(data=data)
    return new_df


class Spider(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.utils_datasets import Spider

    spider = Spider()
    datapoints = spider.get()
    ```
    """

    def __init__(
        self, cache_dir: Optional[str] = None, grouped_sql_queries: bool = False
    ) -> None:
        """
        Initializes the Spider dataset class.

        Args:
            cache_dir (Optional[str], optional): The cache dir to download (or
                load) the dataset. Defaults to None, which uses the default dir.
            grouped_sql_queries (bool, optional): If True, will group examples
                that have identical SQL queries on the same db_id. Grouped
                examples will have multiple NLQs (or any other field that is
                different between examples with identical SQLs). Defaults to False.
        """
        super().__init__(HUGGINGFACE_REPO, cache_dir)
        self.grouped_sql_queries = grouped_sql_queries
        self._schema_cache = {}

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "Spider",
            "description": "Yale Semantic Parsing and Text-to-SQL Challenge.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://yale-lily.github.io/spider",
            "formats": ["json"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the Spider dataset from Huggingface, unzips it, and loads the
        train set, dev set, and tables files.
        """
        # Download the dataset from huggingface
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            branch=BRANCH,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_files, [self.dataset_folder, ["data/spider.zip"]])],
        )

        # Load the train json file
        train_df = pd.read_json(self.dataset_folder + "spider/train_spider.json")
        if self.grouped_sql_queries:
            train_df = group_by_sql_query(train_df)
        # Load the dev json file
        dev_df = pd.read_json(self.dataset_folder + "spider/dev.json")
        if self.grouped_sql_queries:
            dev_df = group_by_sql_query(dev_df)
        # Load the tables (schema) json file
        with open(self.dataset_folder + "spider/tables.json", "r") as fp:
            tables_list = json.load(fp)
        tables_dict = {schema_dict["db_id"]: schema_dict for schema_dict in tables_list}

        self.data = {"train": train_df, "dev": dev_df, "tables": tables_dict}

    def _row_to_datapoint(self, row: pd.Series) -> SqlQueryDatapoint:
        # Check if there are multiple NL Queries for this example
        if isinstance(row["question"], list):
            nl_query = row["question"][0]
            nl_query_alt = row["question"][1:]
        else:
            nl_query = row["question"]
            nl_query_alt = None

        return SqlQueryDatapoint(
            nl_query=nl_query,
            nl_query_alt=nl_query_alt,
            sql_query=row["query"],
            db_id=row["db_id"],
            db_path=self.get_db_path(row["db_id"]),
            db_schema=self.get_schema(row["db_id"]),
            # NOTE: The line bellow assumes that Spider is used for Text-to-SQL only
            # In th future we might want to change this method to `_row_to_text_to_sql_datapoint`
            # and create similar methods for different tasks like `row_to_sql_to_text_datapoint.
            ground_truth=row["query"],
        )

    @requires_files
    def get(self) -> Dict[str, List[SqlQueryDatapoint]]:
        """Returns a dictionary with the train and dev splits of the dataset.

        Returns:
            Dict[str, List[SqlQueryDatapoint]]: A dictionary with keys "train" and "dev",
                and the corresponding lists of SqlQueryDatapoint.
        """
        return {
            "train": [
                self._row_to_datapoint(row) for _, row in self.data["train"].iterrows()
            ],
            "dev": [
                self._row_to_datapoint(row) for _, row in self.data["dev"].iterrows()
            ],
        }

    @requires_files
    def get_raw(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary containing the train, dev, and tables json files
        that have been directly loaded as Pandas DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with keys "train" and "dev",
                and the entire corresponding json file loaded as a DataFrame.
        """
        return self.data

    @requires_files
    def _check_db_id(self, db_id: str) -> None:
        """
        Checks if the given db_id is valid.

        Args:
            db_id (str): A database id

        Returns:
            None if the db_id is a valid one else raise a FileNotFound exception
        """
        if not db_id in self.data["tables"]:
            raise FileNotFoundError(f"The requested db_id ({db_id}) does not exist!")

    @requires_files
    def get_schema(self, db_id: str) -> List:
        """Returns the schema of the database named `db_id`.
        Loads the SQLite database and automatically extracts the schema.

        Args:
            db_id (str): The id of the requested database.

        Returns:
            List: The schema as a list of the DB tables.
        """
        # Get cached schema if it exists
        schema = self._schema_cache.get(db_id, None)
        # If schema was not found, then create it and store it in the cache
        if schema is None:
            # Obtain schema automatically from SQLite file
            db_path = self.get_db_path(db_id)
            # NOTE: We might want to experiment with the value parameters in the future
            schema = cache_auto_db_schema.get_schema(
                db_path, sample_size=20, infer_foreign_keys=True
            )

            # Add notes to schema
            self.add_notes_in_schema(schema, db_id)

            self._schema_cache[db_id] = schema

        return schema

    @requires_files
    def get_db_path(self, db_id: str) -> str:
        """
        Returns the database path that corresponds to the given db_id.

        Args:
            db_id (str): A database id that should appear in the tables.json file.
        """
        self._check_db_id(db_id)

        return f"{self.dataset_folder}/spider/database/{db_id}/{db_id}.sqlite"

    @requires_files
    def execute_query(self, sql_query: str, db_id: str) -> pd.DataFrame:
        """
        Returns the result after executing the query in the database

        Args:
            sql_query (str): The query to be executed.
            db_id (str): The id of the database to execute the query.

        Returns:
            pd.DataFrame: The execution result of the given sql_query in the database that
            corresponds to the given db_id.
        """

        self._check_db_id(db_id=db_id)

        # Get the path of the sqlite file
        sqlite_path = self.get_db_path(db_id)

        return execute_sqlite_query(sql_query=sql_query, sqlite_path=sqlite_path)

    @staticmethod
    def lower_case_query(query: str) -> str:
        """
        Lower case the query without changing values that appear in the query.

        Args:
            query (str): A string of the SQL Query to be lower cased.

        Returns:
            str: The SQL query with characters lower cased.
        """
        return lowercase_query(query)

    @requires_files
    def add_notes_in_schema(self, schema: List, db_id: str) -> None:
        """
        Adds notes to the schema of the database that corresponds to the given db_id. It will add as a note more
        descriptive column/table names only if there is significant difference between the original column/table names
        and the more descriptive ones.

        Args:
            schema (List): The schema obtained by the get_schema method.
            db_id (str): The id of the database to get the notes from.
        """
        table_info = self.data["tables"][db_id]
        table_list = table_info["table_names_original"]

        # Obtain table names
        table_notes = dict(
            zip(table_info["table_names_original"], table_info["table_names"])
        )

        # Obtain column names
        columns_per_table = defaultdict(dict)
        for ind in range(1, len(table_info["column_names"]), 1):
            table_ind = table_info["column_names"][ind][0]
            columns_per_table[table_list[table_ind]][
                table_info["column_names_original"][ind][1]
            ] = table_info["column_names"][ind][1]

        def has_significant_difference(original_name, new_name):
            return not original_name.lower().replace(
                "_", " "
            ) == new_name.lower().replace("_", " ")

        # Add the notes if significant differences exist
        for table in schema:
            if table["table_name"] in table_notes:
                table["note"] = (
                    table_notes[table["table_name"]]
                    if has_significant_difference(
                        table["table_name"], table_notes[table["table_name"]]
                    )
                    else None
                )
            for column in table["columns"]:
                if column["column"] in columns_per_table[table["table_name"]]:
                    note_column = columns_per_table[table["table_name"]][
                        column["column"]
                    ]
                    original_column = column["column"]
                    column["note"] = (
                        note_column
                        if has_significant_difference(original_column, note_column)
                        else None
                    )


if __name__ == "__main__":
    spider = Spider()

    spider_datapoints = spider.get()
    # spider_raw = spider.get_raw()
