from typing import Optional, Dict
import pandas as pd

from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_files

HUGGINGFACE_REPO = "DARELab/imdb"


class IMDB(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.utils_datasets import IMDB

    imdb = IMDB()
    data = imdb.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "IMDB",
            "description": "Created in SQLizer(https://dl.acm.org/doi/pdf/10.1145/3133887) and contains queries upon a"
            " database with movies.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://github.com/jkkummerfeld/text2sql-data/",
            "formats": ["json"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the IMDB dataset from Huggingface, unzips it, and loads the
        test set and tables files.
        """
        # Download the dataset from huggingface
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_files, [self.dataset_folder, ["data.zip"]])],
        )

        # Load the data json file
        data_df = pd.read_json(self.dataset_folder + "data/imdb.json")

        # # Load the tables (schema) json file
        # with open(self.dataset_folder + "data/tables.json", "r") as fp:
        #     tables_list = json.load(fp)
        # tables_dict = {schema_dict["db_id"]: schema_dict for schema_dict in tables_list}
        # self.data = {"test": data_df, "tables": tables_dict}
        self.data = {"test": data_df}

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
        # TODO change when tables.json is created
        # if not db_id in self.data["tables"]:
        if db_id != "imdb":
            raise FileNotFoundError(f"The requested db_id ({db_id}) does not exist!")
