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


class OTTQA(Dataset):
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

    


if __name__ == "__main__":
    ottqa = OTTQA()

    # spider_datapoints = spider.get()
    # spider_raw = spider.get_raw()
