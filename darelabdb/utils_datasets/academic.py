import json
from typing import Optional, Dict
import pandas as pd

from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_files

HUGGINGFACE_REPO = "DARELab/academic"


class Academic(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.utils_datasets import Academic

    academic = Academic()
    data = academic.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "Academic",
            "description": "Created for NaLIR (Li and Jagadish, 2014) by enumerating all of the different queries possible with the Microsoft Academic Search interface, then writing questions for each query.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://github.com/jkkummerfeld/text2sql-data/",
            "formats": ["json"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the Academic dataset from Huggingface, unzips it, and loads the
        test set and tables files.
        """
        # Download the dataset from huggingface
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_files, [self.dataset_folder, ["data.zip"]])],
        )

        # Load the data json file
        data_df = pd.read_json(self.dataset_folder + "data/academic.json")

        # Load the tables (schema) json file
        with open(self.dataset_folder + "data/tables.json", "r") as fp:
            tables_list = json.load(fp)
        tables_dict = {schema_dict["db_id"]: schema_dict for schema_dict in tables_list}

        self.data = {"test": data_df, "tables": tables_dict}

    @requires_files
    def get(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary with the test split of the dataset,
        containing only the NL Question, SQL query and db_id.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with key "test",
                and the corresponding DataFrame with columns "nl_query",
                "sql_query", and "db_id".
        """
        return {
            "test": self.data["test"][["nl_query", "sql_query", "db_id"]],
        }

    @requires_files
    def get_raw(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary containing the test and tables json files
        that have been directly loaded as Pandas DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with key "test",
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
    def get_schema(self, db_id: str) -> pd.DataFrame:
        """Returns the schema json object that corresponds to the given db_id.

        Args:
            db_id (str): A database id that should appear in the tables.json file.

        Returns:
            pd.DataFrame: The json object in the tables.json file that corresponds
                to the given db_id.
        """
        return self.data["tables"][db_id]


if __name__ == "__main__":
    academic = Academic()
    data = academic.get()
