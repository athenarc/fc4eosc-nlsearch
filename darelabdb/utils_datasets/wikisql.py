from typing import Dict, Optional

import pandas as pd
from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.repositories.github import get_files_from_github_repo
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_tarfile

GITHUB_REPO = "salesforce/wikisql"
BRANCH = "master"
FILE_PATHS = ["data.tar.bz2"]

_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]


class WikiSQL(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.datasets import WikiSQL

    wikisql = WikiSQL()
    df = wikisql.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(GITHUB_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns the info of the dataset.
        """
        return {
            "name": "wikisql",
            "description": "A large crowd-sourced dataset for developing natural language interfaces for relational databases.",
            "github_repo": GITHUB_REPO,
            "original_url": "https://github.com/salesforce/WikiSQL",
            "formats": ["jsonl"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        # Download the dataset from github repository
        get_files_from_github_repo(
            GITHUB_REPO,
            BRANCH,
            FILE_PATHS,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_tarfile, [self.dataset_folder, ["data.tar.bz2"]])],
        )

        # Load the files
        self.data = {
            "train": pd.read_json(self.dataset_folder + "data/train.jsonl", lines=True),
            "train_tables": pd.read_json(
                self.dataset_folder + "data/train.tables.jsonl", lines=True
            ),
            "val": pd.read_json(self.dataset_folder + "data/dev.jsonl", lines=True),
            "val_tables": pd.read_json(
                self.dataset_folder + "data/dev.tables.jsonl", lines=True
            ),
            "test": pd.read_json(self.dataset_folder + "data/test.jsonl", lines=True),
            "test_tables": pd.read_json(
                self.dataset_folder + "data/test.tables.jsonl", lines=True
            ),
        }

    @requires_files
    def get_raw(self) -> Dict[str, pd.DataFrame]:
        """Returns a dictionary with the train, validation, and test splits and tables of the dataset."""
        return self.data

    @requires_files
    def get(self, human_readable: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary with the train, validation, and test splits containing only the table_id, question, and sql fields,
            along with tables of the dataset containing only the id, and header fields.

        Args:
            human_readable (bool, optional): If set to True the SQL field will be trasnformed to a readable string instead
                of a dictionary. Defaults to False.
        """

        data = {}

        for key, df in self.data.items():
            if "tables" in key:
                # If this is a dataframe with tables
                data[key] = df[["id", "header"]]
            else:
                # If this is dataframe with NLQ/SQL pairs
                data[key] = df[["table_id", "question", "sql"]]

                if human_readable:
                    # Create a temp index to get tables from ids
                    index_df = self.data[f"{key}_tables"].set_index("id", inplace=False)
                    # Create readable SQL queries
                    data[key]["sql"] = df.apply(
                        lambda x: self._convert_to_human_readable(
                            sel=x["sql"]["sel"],
                            agg=x["sql"]["agg"],
                            columns=index_df.loc[x["table_id"]]["header"],
                            conditions=x["sql"]["conds"],
                        ),
                        axis=1,
                    )

        return data

    def _convert_to_human_readable(self, sel, agg, columns, conditions):
        """Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""

        rep = str(
            f"SELECT "
            f"{_AGG_OPS[agg]}"
            f"{'(' if agg > 0 else ''}"
            f"{columns[sel] if columns is not None else f'col{sel}'}"
            f"{') ' if agg > 0 else ' '}"
            f"FROM table"
        )

        if conditions:
            rep += " WHERE " + " AND ".join(
                [f"{columns[i]} {_COND_OPS[o]} {v}" for i, o, v in conditions]
            )
        return " ".join(rep.split())
