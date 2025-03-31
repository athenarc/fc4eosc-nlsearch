from typing import Type
from darelabdb.utils_datasets import (
    Iris,
    QR2TBenchmark,
    Spider,
    ToTTo,
    WikiSQL,
    Wikitable,
    Bird,
    Fc4eosc,
    Nestle
)
from darelabdb.utils_datasets.dataset_abc import Dataset

SUPPORTED_DATASETS = {
    Iris.__name__: Iris,
    QR2TBenchmark.__name__: QR2TBenchmark,
    Spider.__name__: Spider,
    ToTTo.__name__: ToTTo,
    WikiSQL.__name__: WikiSQL,
    Wikitable.__name__: Wikitable,
    Bird.__name__: Bird,
    Fc4eosc.__name__: Fc4eosc,
    Nestle.__name__: Nestle
}


def dataset_class_picker(dataset_name: str) -> Type[Dataset]:
    """
    Returns a dataset class based on its name.

    Args:
        dataset_name (str): The name of the requested dataset class.

    Returns (type[Dataset]): The Dataset class

    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError(
            f"The dataset {dataset_name} is not implemented. The supported datasets "
            f"are: {list(SUPPORTED_DATASETS.keys())}"
        )

    return SUPPORTED_DATASETS[dataset_name]
