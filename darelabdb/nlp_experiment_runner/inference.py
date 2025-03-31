from typing import List

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_models.seq_to_seq import Seq2SeqModel
from darelabdb.nlp_data_processing.sequential_processing_executor import (
    SequentialProcessingExecutor,
)
from darelabdb.nlp_data_processing.data_module_class import QueryDataModule


def inference(
    model: Seq2SeqModel,
    preprocessor: SequentialProcessingExecutor,
    datapoint: SqlQueryDatapoint,
) -> SqlQueryDatapoint:
    """
    This function can be used to perform inference using a Seq2Seq model.
    It is responsible for handling all steps starting from a datapoint,
    up to returning the final prediction. Specifically, (i) it takes a datapoint,
    (ii) it creates a temporary dataloader that handles pre-processing and tokenizing
    the datapoint, (iii) it uses the `predict_step`  of the model to generate a
    prediction (with post-processing), and (uv) it returns the datapoint containing
    the prediction.

    Args:
        model (Seq2SeqModel): The model to be used for inference.
        preprocessor (SequentialProcessingExecutor): The preprocessor that
            corresponds to the given model.
        datapoint (SqlQueryDatapoint): The input datapoint.

    Returns:
        SqlQueryDatapoint: The datapoint updated with the prediction,
    """
    # Use query datamodule for pre-processing and tokenizing
    dataset = {"train": None, "val": None, "test": [datapoint]}

    dataloader = QueryDataModule(
        dataset,
        1,
        model.tokenizer,
        preprocessor,
        eager=True,
        decoder_only=model.decoder_only,
    ).test_dataloader()

    batch = next(iter(dataloader))

    # Make predictions with the model
    datapoints = model.predict_step(batch)
    outputs = datapoints[0]

    return outputs
