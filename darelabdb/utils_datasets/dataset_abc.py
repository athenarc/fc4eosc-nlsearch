from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union


class Dataset(ABC):
    """
    Set of methods that every dataset should implement.

    Apart from these methods the class can
    implement whichever other method is considered needed e.g. different processed versions of the dataset.
    """

    def __init__(self, huggingface_repo: str, cache_dir: Union[str, None]) -> None:
        """
        Initializer chooses the cache directory where the dataset will be downloaded or retrieved.
        Note the dataset is not yet downloaded.

        Args:
            huggingface_repo: Name of the huggingface repo (i.e. MikeXydas/iris).
            cache_dir: Folder where the dataset will be downloaded or retrieved from.
        """
        self.data = None
        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else str(Path.home()) + "/.cache/darelabdb/datasets/"
        )
        self.huggingface_repo = huggingface_repo
        self.dataset_folder = self.cache_dir + huggingface_repo.split("/")[-1] + "/"

    @abstractmethod
    def _init_data(self):
        """
        Loads the dataset from disk. If the dataset is not available on disk, it is downloaded and saved in cache.
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary with information about the dataset.
        Necessary:
            - **name**: name of dataset
            - **description**: description of dataset
            - **url**: url of gdrive file
            - **original_url**: original url of dataset (can be the paper url)
            - **formats**: list of formats the dataset is available eg. ["csv", "json", "xml"]

        """
        pass

    @abstractmethod
    def get_raw(self):
        """
        Returns the raw data of the dataset on whichever format we consider default.
        Structure (not enforced):
        ```python
            {
                 "train":[],
                 "dev":[],
                 "test":[]
            }
        ```
        """
        pass

    @abstractmethod
    def get(self, **kwargs):
        """
        Returns the processed data of the dataset on whichever format we consider default.
        If many processed versions are available, the one we consider "more" default should be the one returned.
        Structure (not enforced):
        ```python
            {
                 "train":[],
                 "dev":[],
                 "test":[]
            }
        ```
        """
        pass
