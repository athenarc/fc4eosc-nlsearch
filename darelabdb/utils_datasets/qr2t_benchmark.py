import glob
import json
import os
from typing import Dict, Optional

from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.processing.qr2t_benchmark.qr2t_to_totto import to_totto
from darelabdb.utils_datasets.processing.qr2t_benchmark.totto_to_concise_input import (
    to_compact,
)
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files

HUGGINGFACE_REPO = "MikeXydas/qr2t_benchmark"


class QR2TBenchmark(Dataset):
    """
    **Usage**
    ```python
    from darelabdb.datasets import QR2TBenchmark

    qr2t_benchmark = QR2TBenchmark()
    serialized_qr2t = qr2t_benchmark.get()
    ```
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "QR2T Benchmark",
            "description": "QR2T Benchmark is an extension of the Spider dataset for the QR2T task.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "-",
            "formats": ["json"],
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO, cache_dir=self.cache_dir
        )

        self.data = {}
        for file in glob.glob(self.dataset_folder + "*"):
            self.data[os.path.basename(file).split(".")[0]] = json.load(open(file))

    @requires_files
    def get_raw(self):
        """
        Returns a dictionary containing `train`, `dev`, and `eval` splits.
        Each datapoint is a separate dictionary with the following keys:


        ```
        {
            "table_id": "staff",
            "query": "SELECT last_name, email_address FROM staff WHERE email_address LIKE '%wrau%'",
            "table_name": "staff",
            "query_description": "Find the last name of the staff whose email address contains \"wrau\".",
            "results_description": [
                "The staff member whose last name is Erdman has the following email: wrau@example.com",
                "The staff with last name Erdman has the email address wrau@example.com.",
                "The last name is Erdman and the email address is wrau@example.com"
            ],
            "result": ",last_name,email_address\\n0,Erdman,wrau@example.com\\n",
            "difficulty": "small_select"
        }
        ```
        """
        return self.data

    @requires_files
    def get(self):
        """
        Returns a dictionary containing `train`, `dev`, and `eval` splits.
        Each datapoint is a dictionary containing the serialized input of the query results and the verbalisations.

        !!! warning "Train vs. Dev and Evaluation splits"

            The `train` split contains a single verbalisation (`string`) per datapoint while the `dev` and `eval` splits
            contain a list of possible verbalisations (`list[string]`) per datapoint as shown below.

        ```
        {
            'subtable_and_metadata': '<query> club_rank <table> club_rank <col0> Total | NUMERIC | 6 <col1> count of club_rank | NUMERIC | 2 ',
            'final_sentence': ['Two clubs have six medals.', 'There are 2 clubs with 6 medals', 'There are 2 clubs with 6 total medals.']}
        }
        ```
        """
        return {key: to_compact(to_totto(data)) for key, data in self.data.items()}
