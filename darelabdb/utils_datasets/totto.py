import json
from typing import Any, Dict, List, Optional

from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.processing.totto.totto_to_serialized import (
    generate_serialized_table,
)
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_files

HUGGINGFACE_REPO = "MikeXydas/ToTTo"


class ToTTo(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.datasets import ToTTo

    totto = ToTTo()
    data = totto.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "ToTTo",
            "description": "An open-domain English table-to-text dataset.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "https://github.com/google-research-datasets/ToTTo",
            "formats": ["json"],  # List of available formats
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the ToTTo dataset from Huggingface, unzips it, and loads the
        train set, dev set, and tables files.
        """
        # Download the dataset from huggingface
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_files, [self.dataset_folder, ["totto_data.zip"]])],
        )

        def read_json_lines_file(file_path: str) -> List[Dict[str, Any]]:
            with open(file_path, "r") as f:
                return [json.loads(line) for line in f]

        # Load the train, dev, test files
        train = read_json_lines_file(
            self.dataset_folder + "totto_data/totto_train_data.jsonl"
        )
        dev = read_json_lines_file(
            self.dataset_folder + "totto_data/totto_dev_data.jsonl"
        )
        test = read_json_lines_file(
            self.dataset_folder + "totto_data/unlabeled_totto_test_data.jsonl"
        )

        self.data = {"train": train, "dev": dev, "test": test}

    @staticmethod
    def _get_serialized_table(
        datapoint: Dict[str, Any], full_table=False, with_metadata=True
    ) -> Dict[str, str]:
        serialized_table = generate_serialized_table(
            datapoint, with_metadata=with_metadata, full_table=full_table
        )
        if "sentence_annotations" in datapoint:
            final_sentence = [
                annotation["final_sentence"]
                for annotation in datapoint["sentence_annotations"]
            ]
        else:
            final_sentence = None

        return {
            "subtable_and_metadata": serialized_table,
            "final_sentence": final_sentence,
        }

    @requires_files
    def get(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Returns a dictionary with the train and dev splits of the dataset and serialized tables,
        in the format as described in the repo: https://github.com/google-research/language/tree/master/language/totto

        The serialized table only includes the **highlighted cells not the full table**

        **The train split has a single annotation per datapoint, the dev split has multiple,
         and the test split has none.**

         Example returned value:
         ```python
            {
                "train": [{
                    "subtable_and_metadata": "<page_title> title </page_title> <section_title> section </section_title>
                                              <table>
                                                 <cell> value 1 <col_header> col 1 </col_header> </cell>
                                                 <cell> value 2 <col_header> col 2 </col_header> </cell>
                                              </table>",
                    "final_sentence": ["sentence 1"]
                    }]
                "dev": [...]
                "test": [...]
         ```
        """

        return {
            "train": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=False
                )
                for datapoint in self.data["train"]
            ],
            "dev": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=False
                )
                for datapoint in self.data["dev"]
            ],
            "test": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=False
                )
                for datapoint in self.data["test"]
            ],
        }

    @requires_files
    def get_raw(self):
        """
        Returns the raw data as a dictionary with the train and dev splits of the dataset.

        You can check the returned value in: https://github.com/google-research-datasets/ToTTo#dataset-description
        """
        return self.data

    @requires_files
    def get_with_full_table(self):
        """
        Returns a dictionary with the train and dev splits of the dataset and serialized tables,
        in the format as described in the repo: https://github.com/google-research/language/tree/master/language/totto

        The serialized table includes the **highlighted cells AND the full table**

        **The train split has a single annotation per datapoint, the dev split has multiple,
         and the test split has none.**

         Example returned value:
         ```python
            {
                "train": [{
                    "subtable_and_metadata": "<page_title> title </page_title> <section_title> section </section_title>
                                              <table>
                                                 <cell> value 1 <col_header> col 1 </col_header> </cell>
                                                 <cell> value 2 <col_header> col 2 </col_header> </cell>
                                              </table>",
                    "final_sentence": ["sentence 1"]
                    }]
                "dev": [...]
                "test": [...]
         ```
        """
        return {
            "train": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=True
                )
                for datapoint in self.data["train"]
            ],
            "dev": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=True
                )
                for datapoint in self.data["dev"]
            ],
            "test": [
                self._get_serialized_table(
                    datapoint, with_metadata=True, full_table=True
                )
                for datapoint in self.data["test"]
            ],
        }
