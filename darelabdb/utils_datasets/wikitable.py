from typing import Dict, Optional

import pandas as pd
from darelabdb.utils_datasets.dataset_abc import Dataset
from darelabdb.utils_datasets.processing.wikidata import simplify_schema
from darelabdb.utils_datasets.repositories.huggingface import (
    get_file_from_huggingface_hub,
)
from darelabdb.utils_datasets.utils.require_files import requires_files
from darelabdb.utils_datasets.utils.unzip import unzip_files

HUGGINGFACE_REPO = "MikeXydas/wikitable"


class Wikitable(Dataset):
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(HUGGINGFACE_REPO, cache_dir)

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "WIKITABLE",
            "description": "A collection of wikipedia tables.",
            "huggingface_repo": HUGGINGFACE_REPO,
            "original_url": "http://websail-fe.cs.northwestern.edu/TabEL",
            "formats": ["json"],
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }

    def _init_data(self):
        """
        Downloads the Wikitable dataset from Huggingface, unzips it, and loads a reader on the json file.
        The extracted data can be found in the directory named wikitable_unzipped under the dataset_folder.
        """
        # Download the dataset from huggingface
        get_file_from_huggingface_hub(
            repo_id=HUGGINGFACE_REPO,
            cache_dir=self.cache_dir,
            post_processing=[(unzip_files, [self.dataset_folder, ["tables.json.gz"]])],
        )

        self.data = pd.read_json(
            self.dataset_folder + "tables.json", lines=True, chunksize=10000
        )

    @requires_files
    def get_raw(self):
        """
        Returns an **iterator** over the file. The iterator will hang for a bit when loading
        the next chunk.

        Available columns of each row are:
        ```
        > print(next(datapoints))
        {
          "_id": "1000006-1",
          "numCols": 4,
          "numDataRows": 21,
          "numHeaderRows": 1,
          "numericColumns": [],
          "order": 0.8567686371970921,
          "pgId": 1000006,
          "pgTitle": "Römer (crater)",
          "sectionTitle": "Satellite craters",
          "tableCaption": "Satellite craters",
          "tableData": [
            [
              {
                "cellID": -1,
                "textTokens": [],
                "text": "A",
                "tdHtmlString": "<td colspan=\"1\" rowspan=\"1\"> A </td>",
                "surfaceLinks": [],
                "subtableID": -1,
                "isNumeric": false
              },
              ...  # More columns
            ],
            ...  # More rows
          ],
          "tableHeaders": [
            [
              {
                "cellID": -1,
                "textTokens": [],
                "text": "Römer",
                "tdHtmlString": "<th colspan=\"1\" rowspan=\"1\"> Römer </th>",
                "surfaceLinks": [],
                "subtableID": -1,
                "isNumeric": false
              },
              ... # More headers
            ]
          ],
          "tableId": 1
        }
        ```
        """
        self._init_data()  # TODO: This is a hack to make sure that we restart the iterator
        for chunk in self.data:
            for ind, row in chunk.iterrows():
                yield row.to_dict()

    @requires_files
    def get(self):
        """
        Returns an **iterator** over the file. The iterator will hang for a bit when loading the next chunk.
        Each point is reduced to a simplified schema as shown below, which is easier to work with.

        ```
        > print(next(datapoints))
        {
            "table_id": 1,
            "page_title": "movies",
            "section_title": "Quentin Tarantino",
            "caption": "Quentin Tarantino",
            "columns": ["Year", "Title"],
            "rows": [
                ["1992", "Reservoir Dogs"],
                ["1994", "Pulp Fiction"],
                ...
            ]
        }
        ```
        """
        self._init_data()  # TODO: This is a hack to make sure that we restart the iterator
        for chunk in self.data:
            for ind, row in chunk.iterrows():
                yield simplify_schema.simplify(row.to_dict())

    @requires_files
    def load_file_with_custom_loader(self, loader, args, kwargs):
        """
        Returns the contents of the whole json file using the loader function with the given arguments.
        Example usage: `dataset.custom_loader_of_whole_file(polars.read_json, [], {})`

        !!! tip "Loading huge files and Polars 🐻‍❄️"

            The unzipped json file is around 70GB and pandas will find it difficult to load it and analyze it.
            We suggest using [Polars](https://github.com/pola-rs/polars) 🐻‍❄️ instead,
            which specializes in huge datasets.

        Args:
            loader (Any): The loader function to use.
            args (List[Any]): The arguments to pass to the loader function.
            kwargs (Dict[Any]): The keyword arguments to pass to the loader function.

        Returns:
            Any: The contents of the whole json file using the loader function with the given arguments.
        """
        return loader(self.dataset_folder + "tables.json", *args, **kwargs)
