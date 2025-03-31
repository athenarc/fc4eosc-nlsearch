from typing import Callable, Dict, Iterator, Optional, Union

import lightning as pl
from darelabdb.nlp_data_processing.data_module_class import QueryDataModule
from darelabdb.nlp_data_processing.dataset_adapters.spider_adapter import (
    spider_to_sql_query_datapoints,
)
from darelabdb.nlp_data_processing.dataset_adapters.bird_adapter import (
    bird_to_sql_query_datapoints,
)
from darelabdb.utils_datasets import Spider, Bird
from darelabdb.utils_datasets.dataset_abc import Dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase


def to_data_module(
    dataset: Dataset,
    processor,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    shuffle: bool = False,
    decoder_only: bool = False,
    eager: bool = False,
    cache_dir_name: Optional[str] = None,
    max_length: int = 256,
) -> pl.LightningDataModule:
    """
    Converts a dataset to a data module.

    Args:
        dataset: A darelabdb.datasets.dataset_abc.Dataset object.
        processor: A processor object.
        batch_size: The batch size of the data module for all splits.
        tokenizer: The tokenizer used for the text-to-text model.
        shuffle: If set to True the examples of the train set are shuffled after
            each epoch
        decoder_only: Set to True if model's architecture is decoder-only,
            set to False if architecture is encoder-decoder
        eager: If True, the data module will do all the preprocessing before the train stage.
        cache_dir_name: A unique identifier of the processed dataset that can be used to cache it for future runs.
            If None the dataset will not be stored. Requires eager to be True.
        max_length: The max length of the encoded texts from the tokenizer.
            Defaults to 256.

    Returns:
        A pytorch lightning data module.
    """
    if not eager and cache_dir_name is not None:
        logger.error(
            "The dataset will not be cached since eager is False, even if cache_dir_name is not None. "
            "eager should be True if you want to cache the dataset"
        )
        cache_dir_name = None

    return QueryDataModule(
        dataset=dataset_adapter_picker(dataset)(dataset),
        batch_size=batch_size,
        tokenizer=tokenizer,
        processor=processor,
        eager=eager,
        cache_dir_name=cache_dir_name,
        decoder_only=decoder_only,
        shuffle=shuffle,
        max_length=max_length,
    )


def dataset_adapter_picker(
    dataset: Dataset,
) -> Callable[[Union[Spider, Bird]], Dict[str, Iterator]]:
    if isinstance(dataset, Spider):
        return spider_to_sql_query_datapoints
    if isinstance(dataset, Bird):
        return bird_to_sql_query_datapoints
    else:
        raise NotImplementedError(
            f"Data module for dataset {dataset.__class__.__name__} is not implemented"
        )
