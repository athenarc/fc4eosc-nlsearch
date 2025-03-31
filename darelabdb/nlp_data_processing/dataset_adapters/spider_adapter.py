from typing import Dict, Iterator

import pandas as pd
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.utils_datasets import Spider


def spider_to_sql_query_datapoints(
    spider_dataset: Spider,
) -> Dict[str, Iterator[SqlQueryDatapoint]]:
    """
    Converts a Spider dataset to a QueryDatapoint dataset.

    Args:
        spider_dataset: A Spider dataset object.

    Returns:
        A dictionary with keys "train", "val", and "test" and the corresponding generators of QueryDatapoint objects.
    """
    data = spider_dataset.get_raw()

    train = data["train"]
    val = data["dev"]

    return {
        "train": create_sql_datapoint_generator(train, spider_dataset),
        "val": create_sql_datapoint_generator(val, spider_dataset),
        "test": None,
    }


def create_sql_datapoint_generator(split: pd.DataFrame, dataset: Spider):
    for _, row in split.iterrows():
        yield dataset._row_to_datapoint(row)
