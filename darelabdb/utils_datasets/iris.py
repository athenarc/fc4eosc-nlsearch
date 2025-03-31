import typing

import numpy as np  # For train/test split
import pandas as pd  # For reading the iris csv file
from darelabdb.utils_datasets.dataset_abc import Dataset  # The dataset abstract class
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,  # Download from huggingface
)
from darelabdb.utils_datasets.utils.require_files import (
    requires_files,  # Decorator dealing with dataset caching
)

HUGGINGFACE_REPO = "MikeXydas/iris"


class Iris(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.datasets import Iris

    iris = Iris()
    df = iris.get()
    ```
    """

    def __init__(self, cache_dir: typing.Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> typing.Dict[str, str]:
        """
        Returns the info of the dataset.
        """
        return {
            "name": "iris",
            "description": "Three species of iris plants.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://www.kaggle.com/datasets/uciml/iris?resource=download",
            "formats": ["csv"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        # Download the dataset from huggingface dataset repository
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO, cache_dir=self.cache_dir
        )

        # Load the csv file
        self.data = pd.read_csv(self.dataset_folder + "iris.csv")

    @requires_files
    def get_raw(self) -> pd.DataFrame:
        """Returns the raw csv file of all the plants"""
        return self.data

    @requires_files
    def get(self, train_size: float = 0.8) -> typing.Dict[str, pd.DataFrame]:
        """
        Returns a dictionary with the train and test splits of the dataset.

        Args:
            train_size: The size of the train split. Defaults to 0.8.
        """
        mask = np.random.rand(len(self.data)) < train_size

        return {"train": self.data[mask], "test": self.data[~mask]}
