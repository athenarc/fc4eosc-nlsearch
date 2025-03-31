from typing import Dict, Iterator

import pandas as pd
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.utils_datasets import Bird


def bird_to_sql_query_datapoints(
    bird_dataset: Bird,
) -> Dict[str, Iterator[SqlQueryDatapoint]]:
    """
    Converts a Bird dataset to a QueryDatapoint dataset.

    Args:
        bird_dataset: A Bird dataset object.

    Returns:
        A dictionary with keys "train", "val", and "test" and the corresponding
            generators of QueryDatapoint objects.
    """
    data = bird_dataset.get_raw()

    train = data["train"]
    val = data["dev"]

    return {
        "train": create_sql_datapoint_generator(train, bird_dataset),
        "val": create_sql_datapoint_generator(val, bird_dataset),
        "test": None,
    }


def create_sql_datapoint_generator(split: pd.DataFrame, dataset: Bird):
    for _, row in split.iterrows():
        yield dataset._row_to_datapoint(row)
