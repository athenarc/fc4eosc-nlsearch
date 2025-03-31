import os
from typing import Dict, Iterator, List, Optional, Callable

import lightning as pl
import pandas as pd
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from loguru import logger
from torch.utils.data import (
    Dataset as TorchDataset,  # This is ugly but might help avoid confusion with our Dataset class
    DataLoader,
    default_collate,
)
from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq
from tqdm.auto import tqdm


class CustomCollator:
    def __init__(self, collate_fn: Callable = default_collate) -> None:
        self.collate_fn = collate_fn

    def collate(self, batch: Dict) -> Dict:
        """
        A custom collate function that temporarily removes datapoints from the
        batch because the default collator can not handle them.

        Args:
            batch (Dict): The batch to be collated

        Returns:
            Dict: The collated batch
        """
        # Remove datapoints so that they are not collated
        datapoints = [example.pop("datapoints") for example in batch]

        # Perform default collate on the batch with the datapoints
        collated_batch = self.collate_fn(batch)
        # Transform batch to standard dict, to avoid lightning trying to place datapoints in gpu
        collated_batch = dict(collated_batch)

        # Add datapoints back after batch has been collated
        collated_batch["datapoints"] = datapoints

        return collated_batch


class MockProcessor:
    @staticmethod
    def process(query_datapoint):
        return query_datapoint


class QueryDataset(TorchDataset):
    def __init__(
        self,
        datapoints: List[SqlQueryDatapoint],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        decoder_only: bool = False,
        val_mode: bool = False,
    ) -> None:
        """
        A Dataset class that handles tokenizing the inputs and labels for
        a language model.

        Args:
            datapoints (List[SqlQueryDatapoint]): The list of datapoints that
                form the dataset.
            tokenizer (PreTrainedTokenizerBase): The tokenizer that will be used
                to tokenize the inputs and labels.
            max_length (int, optional): The max length of the encoded texts from
                the tokenizer. Defaults to 256.
            decoder_only (bool, optional): Whether the model's architecture is
                decoder-only. This will affect the way the labels are tokenized.
                Defaults to False.
            val_mode (bool, optional): If decoder_only is set to True, this argument
                indicates whether the dataset should be tokenized for validation
                or training. This happens because decoder-only models, have the
                label in the input tokens. Defaults to False.
        """
        self.datapoints = datapoints
        self.input_texts = [datapoint.model_input for datapoint in datapoints]
        self.label_texts = [datapoint.expected_output for datapoint in datapoints]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.decoder_only = decoder_only
        self.val_mode = val_mode

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoints = self.datapoints[index]
        inputs = self.input_texts[index]
        labels = self.label_texts[index]

        def template(input, label):
            # TODO: Find a better home for this?
            return (
                f"You must output a natural language explanation of the input SQL query.\n\n"
                f"### Input:\n"
                f"{input}\n\n"
                f"### Response:\n"
                f"{label}"
            )

        if self.decoder_only:
            # Prepare tokens for decoder-only architecture
            if isinstance(index, int):
                # TODO: Write this dataset based on the assertion that items are indexed one at a time
                # TODO: Document above assertion
                inputs = [inputs]
                labels = [labels]

            prompts_no_labels = [template(input, "") for input in inputs]

            encodings_dict_no_labels = self.tokenizer(
                text=prompts_no_labels,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            if self.val_mode:
                encodings_dict = encodings_dict_no_labels
            else:
                prompts_with_labels = [
                    template(input, label) + f"{self.tokenizer.eos_token}"
                    for input, label in zip(inputs, labels)
                ]
                encodings_dict = self.tokenizer(
                    text=prompts_with_labels,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

                encodings_dict["labels"] = encodings_dict["input_ids"].clone()
                # Mask the part of the labels that are not in the target output
                encodings_dict["labels"][0][
                    : len(encodings_dict_no_labels["input_ids"][0])
                ] = -100

        else:
            # Prepare tokens for encoder-decoder architecture
            encodings_dict = self.tokenizer(
                text=inputs,
                text_target=labels,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

        # Flatten to one dimension because it's handled by DataLoader?
        encodings_dict = {k: v.flatten() for k, v in encodings_dict.items()}

        # Add datapoints to the dict
        encodings_dict["datapoints"] = datapoints

        return encodings_dict


class QueryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Iterator],
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        processor: Optional[MockProcessor] = None,
        eager: bool = False,
        cache_dir_name: Optional[str] = None,
        max_length: int = 256,
        decoder_only: bool = False,
        shuffle: bool = False,
    ):
        super().__init__()
        self.dataset: Dict[str, Iterator] = dataset
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.processor: Optional[MockProcessor] = processor
        self.batch_size: int = batch_size
        self.eager: bool = eager
        self.max_length = max_length
        self.decoder_only: bool = decoder_only
        self.shuffle: bool = shuffle

        # Initialise collator for batching data
        seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, max_length=self.max_length, padding="longest"
        )
        self.collator = CustomCollator(collate_fn=seq2seq_collator)

        if eager is True and cache_dir_name is None:  # Eager but no caching
            self.train, self.val, self.test = self._get_processed_data()
        elif eager is True and cache_dir_name is not None:  # Eager and caching
            self.train, self.val, self.test = self.get_cached_dataset(cache_dir_name)
        else:  # Not eager
            self.train, self.val, self.test = None, None, None

    def prepare_data(self) -> None:
        """Dataset is already downloaded"""
        pass

    def _get_processed_data(self):
        if self.processor is None:
            # If there is no pre-processor, just return the original dataset
            return (
                self.dataset["train"] if self.dataset["train"] is not None else None,
                self.dataset["val"] if self.dataset["val"] is not None else None,
                self.dataset["test"] if self.dataset["test"] is not None else None,
            )
        else:
            processed_datasets = {}
            for split in ["train", "val", "test"]:
                # For each split of the dataset that is not None, pre-process each datapoint
                if self.dataset[split] is None:
                    processed_datasets[split] = None
                else:
                    logger.info(f"Preprocessing {split} dataset...")
                    processed_datasets[split] = [
                        self.processor.process(datapoint)
                        for datapoint in tqdm(self.dataset[split])
                    ]

            return (
                processed_datasets["train"],
                processed_datasets["val"],
                processed_datasets["test"],
            )

    def setup(self, stage: str) -> None:
        # If the data are not loaded
        if not self.eager:
            self.train, self.val, self.test = self._get_processed_data()

    def _get_dataloader(
        self, dataset: List[SqlQueryDatapoint], val_mode=False
    ) -> DataLoader:
        dataset = QueryDataset(
            dataset,
            self.tokenizer,
            self.max_length,
            decoder_only=self.decoder_only,
            val_mode=val_mode,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and not val_mode,
            collate_fn=self.collator.collate,
        )

    def train_dataloader(self):
        if self.train is None:
            return None

        return self._get_dataloader(self.train)

    def val_dataloader(self):
        if self.val is None:
            return None

        return self._get_dataloader(self.val, val_mode=True)

    def test_dataloader(self):
        if self.test is None:
            return None

        return self._get_dataloader(self.test, val_mode=True)

    def predict_dataloader(self):
        if not self.test is None:
            subset = self.test
        elif not self.val is None:
            subset = self.val
        else:
            return None

        return self._get_dataloader(subset, val_mode=True)

    def get_cached_dataset(self, cache_dir_name):
        if cache_dir_name[-1] != "/":
            cache_dir_name += "/"

        if os.path.isdir(cache_dir_name):
            logger.info(f"Using the cached version found in {cache_dir_name}")
            train = (
                pd.read_csv(cache_dir_name + "train.csv")
                if os.path.exists(cache_dir_name + "train.csv")
                else None
            )
            val = (
                pd.read_csv(cache_dir_name + "val.csv")
                if os.path.exists(cache_dir_name + "val.csv")
                else None
            )
            test = (
                pd.read_csv(cache_dir_name + "test.csv")
                if os.path.exists(cache_dir_name + "test.csv")
                else None
            )
        else:
            train, val, test = self._get_processed_data()
            os.mkdir(cache_dir_name)

            # This should be changed. We should establish a common representation of the processor output.
            # Then we should have an idea of how to save it (and load it above).
            if train is not None:
                pd.DataFrame([dict(s) for s in train]).to_csv(
                    cache_dir_name + "train.csv", index=False
                )
            if val is not None:
                pd.DataFrame([dict(s) for s in val]).to_csv(
                    cache_dir_name + "val.csv", index=False
                )
            if test is not None:
                pd.DataFrame([dict(s) for s in test]).to_csv(
                    cache_dir_name + "test.csv", index=False
                )
            logger.info(f"Created dataset cached version in {cache_dir_name}")

        return train, val, test
