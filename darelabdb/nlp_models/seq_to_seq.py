from typing import Any, List, Dict, Literal
from abc import ABC, abstractmethod

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Adafactor,
)
from peft import get_peft_model, PeftConfig

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_data_processing.sequential_processing_executor import (
    SequentialProcessingExecutor,
)


class Seq2SeqModel(ABC, L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        decoder_only: bool = False,
        learning_rate: float = 5e-5,
        optimizer: Literal["adamw", "adafactor"] = "adamw",
        postprocessor: SequentialProcessingExecutor = None,
        peft_config: PeftConfig = None,
        max_output_length: int = 512,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        do_sample: bool = False,
    ) -> None:
        """
        An abstract base class for a seq-to-seq model (i.e., takes text as input
        and generates text as output). The children of this have to implement
        the methods that handle the metric calculations.

        Args:
            model_name_or_path (str): The name (to download from huggingface) or
                path (to a downloaded model) of the language model that is used
                by the model.
            decoder_only (bool, optional): If the architecture of the given
                language model is decoder-only (instead of encoder-decoder).
                Defaults to False.
            learning_rate (float, optional): The learning rate for training the
                model. Defaults to 5e-5.
            optimizer (Literal[&quot;adamw&quot;], optional): The optimizer to
                use when training the model. Defaults to "adamw".
            postprocessor (SequentialProcessingExecutor, optional): The postprocessor
                that handles the raw predictions of the language model. Defaults to None.
            peft_config (PeftConfig, optional): If a peft configuration is given
                then it will be applied to the model when it is initialized.
                Defaults to None.
            max_output_length (int, optional): The maximum length limit of the
                model's generated predictions. Defaults to 512.
            num_beams (int, optional): The number of beams to use when generating
                a prediction. Defaults to 4.
            num_return_sequences (int, optional): The number of predictions to
                generate per input. Be aware that if this is greater than 1, then
                the postprocessor must handle the extra prediction. Defaults to 1.
            do_sample (bool, optional): Whether or not to perform sampling when
                generating a prediction. Defaults to False.
        """
        super().__init__()

        self.save_hyperparameters()

        # Set arguments
        self.decoder_only = decoder_only
        self.postprocessor = postprocessor
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # Set generation parameters
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample

        # Initialise metrics
        self._init_metrics()

        # Initialise model based on architecture
        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.decoder_only:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, config=config
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, config=config
            )

        if peft_config:
            # If a configuration for parameter-efficient fine-tuning was passed
            # Then apply it to the model
            self.model = get_peft_model(self.model, peft_config)

        # Initialise tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, config=config
        )
        if self.tokenizer.pad_token_id is None:
            # Some tokenizers do not have a pad token and must be set manually
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.decoder_only:
            # In decoder-only models right-padding helps speed-up predictions
            self.tokenizer.padding_side = "left"

    @abstractmethod
    def _init_metrics(self):
        pass

    @abstractmethod
    def _update_metrics(self, datapoints: List[SqlQueryDatapoint]):
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, batch) -> Any:
        batch.pop("datapoints")

        outputs = self.model(**batch)

        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, logits = self.forward(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def generate_predictions(
        self, input_ids: torch.Tensor, attention_masks: torch.Tensor
    ) -> List[str]:
        # Generate prediction sequences
        generated_outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_output_length,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated_sequences = generated_outputs.sequences

        if self.decoder_only:
            # Decoder-only models keep the input sequence in their predictions
            # Mask out all input tokens
            input_lengths = torch.sum(attention_masks, dim=1)
            for i in range(len(input_lengths)):
                # Some generated sequences will also have padding in their beginning
                pad_ending = 0
                for j in range(len(generated_sequences[i])):
                    if generated_sequences[i][j] != self.tokenizer.pad_token_id:
                        pad_ending = j
                        break
                generated_sequences[i][
                    pad_ending : pad_ending + input_lengths[i]
                ] = self.tokenizer.pad_token_id

        # Decode predictions
        predictions = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        return predictions

    def predict_step(self, batch) -> List[SqlQueryDatapoint]:
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        predictions = self.generate_predictions(input_ids, attention_masks)

        # Add predictions to datapoints
        for i, datapoint in enumerate(batch["datapoints"]):
            if self.num_return_sequences > 1:
                # If there are multiple predictions for the same input, group them
                start_idx = i * self.num_return_sequences
                end_idx = start_idx + self.num_return_sequences
                grouped_predictions = predictions[start_idx:end_idx]

                datapoint.candidates = grouped_predictions
            else:
                datapoint.prediction = predictions[i]

        # Apply post-processing to the predictions
        datapoints = self._postprocess_batch(batch["datapoints"])

        return datapoints

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        # Get the batch predictions
        datapoints = self.predict_step(batch)

        # Logging predictions with WandB
        if batch_idx in [0, 10, 20, 30, 40] and isinstance(self.logger, WandbLogger):
            # Prepare columns and data for logging
            columns = ["epoch", "model_input", "prediction", "ground_truth"]
            data = [
                [
                    self.trainer.current_epoch,
                    datapoint.model_input,
                    datapoint.prediction,
                    datapoint.ground_truth,
                ]
                for datapoint in datapoints
            ]
            # Log predictions
            self.logger.log_table(key="predictions_samples", columns=columns, data=data)

        # Update metrics with datapoints having predictions
        self._update_metrics(datapoints)

        # Return predictions
        predictions = [datapoint.prediction for datapoint in datapoints]
        return predictions

    def on_validation_epoch_end(self):
        metrics_results = self._compute_metrics()
        for metric_name, metric_score in metrics_results.items():
            self.log(
                f"val_{metric_name}",
                metric_score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx)

    def _postprocess_batch(
        self, datapoints: List[SqlQueryDatapoint]
    ) -> List[SqlQueryDatapoint]:
        if self.postprocessor is None:
            # If there is not post-processor then there is nothing to do
            return datapoints
        else:
            processed_datapoints = [
                self.postprocessor.process(datapoint) for datapoint in datapoints
            ]
            return processed_datapoints

    def get_scores(
        self, datapoints: List[SqlQueryDatapoint]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the metric scores for a given set of datapoints. Each
        datapoint must contain the ground truth and the prediction.

        Args:
            datapoints (List[SqlQueryDatapoint]): The datapoints for which we
                want to calculate the metrics.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping the metric name to
                its score for the given datapoints.
        """
        # Add datapoints to the model's metrics
        self._update_metrics(datapoints)
        # Calculate the metrics
        metric_scores = self._compute_metrics()
        return metric_scores

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer == "adafactor":
            return Adafactor(
                self.model.parameters(),
                lr=self.learning_rate,
                relative_step=False,
                # NOTE: Not sure if it's better or not to have them set to False
                # scale_parameter=False,
                # warmup_init=False,
            )
        else:
            raise NotImplementedError(
                f"There requested optimizer ({self.optimizer}) is not supported!"
            )
