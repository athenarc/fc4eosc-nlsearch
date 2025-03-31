import json
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
from loguru import logger

HUGGINGFACE_REPO = "DARELab/BIRD"
BRANCH = "main"


class Bird(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.utils_datasets import Bird

    bird = Bird()
    datapoints = bird.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        Initializes the BIRD dataset class.

        Args:
            cache_dir (Optional[str], optional): The cache dir to download (or
                load) the dataset. Defaults to None, which uses the default dir.
        """
        super().__init__(HUGGINGFACE_REPO, cache_dir)
        self._schema_cache = {}

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "BIRD",
            "description": "Big Bench for Large-Scale Database Grounded Text-to-SQLs.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://yale-lily.github.io/spider",
            "formats": ["json"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the BIRD dataset from Huggingface unzips it, and loads the
        train set, train tables, dev set, and dev tables files.
        """
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            branch=BRANCH,
            cache_dir=self.cache_dir,
            post_processing=[
                (
                    unzip_files,
                    [
                        self.dataset_folder,
                        [
                            "train.zip",
                            "dev.zip",
                            "train/train_databases.zip",
                            "dev/dev_databases.zip",
                        ],
                    ],
                )
            ],
        )

        # Load the train.json file
        train_df = pd.read_json(self.dataset_folder + "train/train.json")
        # Load the train_tables.json file
        with open(self.dataset_folder + "train/train_tables.json", "r") as fp:
            tables_list = json.load(fp)
        train_tables_dict = {
            schema_dict["db_id"]: schema_dict for schema_dict in tables_list
        }

        # Load the dev.json file
        dev_df = pd.read_json(self.dataset_folder + "dev/dev.json")
        # Load the dev_tables.json file
        with open(self.dataset_folder + "dev/dev_tables.json", "r") as fp:
            tables_list = json.load(fp)
        dev_tables_dict = {
            schema_dict["db_id"]: schema_dict for schema_dict in tables_list
        }

        self.data = {
            "train": train_df,
            "train_tables": train_tables_dict,
            "dev": dev_df,
            "dev_tables": dev_tables_dict,
        }

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
            sql_query=row["SQL"],
            db_id=row["db_id"],
            db_path=self.get_db_path(row["db_id"]),
            db_schema=self.get_schema(row["db_id"]),
            # NOTE: The line bellow assumes that bird is used for Text-to-SQL only
            # In th future we might want to change this method to `_row_to_text_to_sql_datapoint`
            # and create similar methods for different tasks like `row_to_sql_to_text_datapoint.
            ground_truth=row["SQL"],
            evidence=row["evidence"],
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
        if not (db_id in self.data["train_tables"] or db_id in self.data["dev_tables"]):
            raise FileNotFoundError(f"The requested db_id ({db_id}) does not exist!")

    @requires_files
    def get_tables(self, db_id: str) -> pd.DataFrame:
        """
        Returns the tables json object that corresponds to the given db_id.

        Args:
            db_id (str): A database id that should appear in the tables.json file.

        Returns:
            pd.DataFrame: The json object in the tables.json file that corresponds
                to the given db_id.
        """
        if db_id in self.data["train_tables"]:
            return self.data["train_tables"][db_id]
        elif db_id in self.data["dev_tables"]:
            return self.data["dev_tables"][db_id]
        else:
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
        if db_id in self.data["train_tables"]:
            return f"{self.dataset_folder}train_databases/{db_id}/{db_id}.sqlite"
        elif db_id in self.data["dev_tables"]:
            return f"{self.dataset_folder}dev_databases/{db_id}/{db_id}.sqlite"
        else:
            raise FileNotFoundError(f"The requested db_id ({db_id}) does not exist!")

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
        # Get the path of the sqlite file
        sqlite_path = self.get_db_path(db_id)

        return execute_sqlite_query(sql_query=sql_query, sqlite_path=sqlite_path)

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
        table_info = self.get_tables(db_id)

        table_notes = dict(
            zip(table_info["table_names_original"], table_info["table_names"])
        )

        db_info_path = (
            "/".join(self.get_db_path(db_id).split("/")[:-1]) + "/database_description/"
        )

        def has_significant_difference(original_name, new_name):
            return not (
                original_name.lower().replace("_", "").replace(" ", "")
                == new_name.lower().replace("_", "").replace(" ", "")
            )

        for table in schema:
            try:
                original_table_name = table["table_name"].strip("`")
                note_table_name = table_notes[table["table_name"]]

                if original_table_name in table_notes:
                    table["note"] = (
                        table_notes[original_table_name]
                        if has_significant_difference(
                            original_table_name, note_table_name
                        )
                        else None
                    )

                    aliases = pd.read_csv(db_info_path + f"{original_table_name}.csv")

                    for column in table["columns"]:
                        original_column_name = column["column"].strip("`")
                        column_description = aliases[
                            aliases["original_column_name"].str.strip()
                            == original_column_name
                        ]["column_name"].iloc[0]

                        if (
                            column_description
                            and type(column_description) == str
                            and column_description != ""
                            and has_significant_difference(
                                original_column_name, column_description
                            )
                        ):
                            column["note"] = column_description
            except (KeyError, UnicodeError, FileNotFoundError) as e:
                logger.warning(
                    f"While adding notes for table {table['table_name']}, the following error occurred: {e}"
                )
                table["note"] = None
                for column in table["columns"]:
                    column["note"] = None


if __name__ == "__main__":
    bird = Bird()
    schema = bird.get_schema("card_games")
    bird.add_notes_in_schema(schema, "card_games")

    print(schema)

    bird_datapoints = bird.get()
    # bird_raw = bird.get_raw()
