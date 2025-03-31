from typing import Dict, List

import torch
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_metrics.execution_accuracy import ExecutionAccuracy
from darelabdb.nlp_models.seq_to_seq import Seq2SeqModel
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb


class Seq2SeqText2SqlModel(Seq2SeqModel):
    def _init_metrics(self):
        # WARNING: Exact match has been disabled for now. If needed we can add it back
        # This is a direct cause of the decision to not use git subcomponents
        # self.exact_match = TestSuiteExactMatch()
        self.execution_accuracy = ExecutionAccuracy()

    def _update_metrics(self, datapoints: List[SqlQueryDatapoint]):
        predictions = [datapoint.prediction for datapoint in datapoints]
        targets = [datapoint.ground_truth for datapoint in datapoints]
        db_paths = [datapoint.db_path for datapoint in datapoints]

        self.exact_match.update(predictions, targets, db_paths)
        self.execution_accuracy.update(predictions, targets, db_paths)

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        metrics_results = {
            self.exact_match.name(): self.exact_match.compute(),
            self.execution_accuracy.name(): self.execution_accuracy.compute(),
        }
        return metrics_results

    def on_fit_start(self):
        # Before starting to fit the model, we might want to set up some things
        if isinstance(self.logger, WandbLogger):
            # If using a WandB logger, specify how to summarize the metric
            wandb.define_metric(f"val_{self.exact_match.name()}", summary="max")
            wandb.define_metric(f"val_{self.execution_accuracy.name()}", summary="max")
