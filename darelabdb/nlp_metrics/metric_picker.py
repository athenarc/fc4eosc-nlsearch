from torchmetrics import Metric

from torchmetrics.text import BLEUScore

from darelabdb.nlp_metrics.execution_accuracy import (
    ExecutionAccuracy,
)

SUPPORTED_METRICS = {
    ExecutionAccuracy.__name__: ExecutionAccuracy,
    BLEUScore.__name__: BLEUScore,
}


def metric_picker(metric_name: str) -> Metric:
    """
    Returns an instance of a metric based on its name.
    Args:
        metric_name (str): The name of the requested metric.

    Returns (Metric): Metric

    """
    if metric_name not in SUPPORTED_METRICS:
        raise NotImplementedError(
            f"The metric {metric_name} is not implemented. The supported metrics "
            f"are: {list(SUPPORTED_METRICS.keys())}"
        )

    return SUPPORTED_METRICS[metric_name]()
