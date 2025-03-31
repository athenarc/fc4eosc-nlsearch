from typing import Dict, List

import pandas as pd
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.db_schema import cache_auto_db_schema
from darelabdb.utils_database_connector.db_schema.filter_schema import filter_schema
from darelabdb.utils_datasets.utils.require_files import requires_files

EXCLUDED_COLUMNS = {}


class Nestle:
    """
    Note: Requires Athena VPN to execute queries.

    **Usage**
    ```python
    from darelabdb.utils_datasets import Nestle

    nestle = Nestle()
    datapoints = nestle.get()
    ```
    """

    def __init__(self) -> None:
        """
        Initializes the Nestle dataset class.
        """
        self.dataset_path = (
            "components/darelabdb/utils_datasets/assets/nestle/nestle_benchmark.xlsx"
        )
        self.database_name = "nestle"
        # self.database_schema = ""
        self.data = None
        self.db = Database(database=self.database_name)
        self.cached_schema = None

    @staticmethod
    def get_info() -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "Nestle",
            "description": "Nestle benchmark dataset created by darelab",
            "original_url": "",
            "formats": ["xlsx"],  # List of available formats
            "dataset_folder": "components/darelabdb/utils_datasets/assets/nestle/nestle_benchmark.xlsx",
        }

    def _init_data(self):
        """
        Reads the benchmark from the excel file and stores it in the data attribute.
        """
        # Download the dataset from huggingface
        self.data = pd.read_excel(self.dataset_path)

    def _row_to_datapoint(self, row: pd.Series) -> SqlQueryDatapoint:
        schema = self.cached_schema if self.cached_schema else self.get_schema()

        return SqlQueryDatapoint(
            nl_query=str(row["nl_question"]),
            sql_query=str(row["sql_query"]),
            db_id=f"{self.database_name}",
            db_path=f"{self.database_name}",
            db_schema=filter_schema(schema, exclude_values=EXCLUDED_COLUMNS),
            ground_truth=str(row["sql_query"]),
        )

    @requires_files
    def get(self) -> List[SqlQueryDatapoint]:
        """Returns a list of datapoints of the benchmark.

        Returns:
            List[SqlQueryDatapoint]: A list of SqlQueryDatapoint.
        """
        return [self._row_to_datapoint(row) for _, row in self.data.iterrows()]

    @requires_files
    def get_raw(self) -> pd.DataFrame:
        """
        Returns a DataFame of the benchmark read

        Returns:
            pd.DataFrame: The DataFrame of the benchmark.
        """
        return self.data

    def get_schema(self) -> List:
        """
        Returns the schema of the nestle database.

        Returns:
            List: The schema as a list of the DB tables.
        """
        # Get cached schema if it exists
        self.cached_schema = cache_auto_db_schema.get_schema(
            database_str=f"{self.database_name}",
            sample_size=4,
            infer_foreign_keys=False,
        )

        return self.cached_schema

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Returns the result after executing the query in the database

        Args:
            sql_query (str): The query to be executed.

        Returns:
            pd.DataFrame: The execution result of the given sql_query in the nestle database.
        """
        return self.db.execute(sql_query, limit=-1)
