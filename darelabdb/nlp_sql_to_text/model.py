from typing import List, Dict

from loguru import logger
import torch
import wandb
from torchmetrics.text import BLEUScore
from lightning.pytorch.loggers import WandbLogger

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_models.seq_to_seq import Seq2SeqModel


class Seq2SeqSql2TextModel(Seq2SeqModel):

    def _init_metrics(self):
        self.bleu_score = BLEUScore()

    def _update_metrics(self, datapoints: List[SqlQueryDatapoint]):
        # Prepare predictions and targets for metrics
        predictions = []
        targets = []
        for datapoint in datapoints:
            predictions.append(datapoint.prediction)

            # NOTE: The BLEU metric expects a list of targets for each
            # prediction because it can support multiple references
            target = [datapoint.ground_truth]
            if datapoint.nl_query_alt:
                # If there are additional NL Queries in this example, add them
                target.extend(datapoint.nl_query_alt)
            targets.append(target)

        # Update BLEU Score
        self.bleu_score.update(predictions, targets)

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        metrics_results = {"BLEUScore": self.bleu_score.compute()}
        return metrics_results

    def on_fit_start(self):
        # Before starting to fit the model, we might want to set up some things
        if isinstance(self.logger, WandbLogger):
            # If using a WandB logger, specify how to summarize the metric
            wandb.define_metric("val_BLEUScore", summary="max")
