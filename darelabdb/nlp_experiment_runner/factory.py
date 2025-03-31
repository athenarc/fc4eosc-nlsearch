from typing import Tuple, Union

from darelabdb.nlp_data_processing.data_module_class import QueryDataModule
from darelabdb.nlp_data_processing.create_data_module import to_data_module
from darelabdb.nlp_metrics.metric_picker import metric_picker
from darelabdb.nlp_sql_to_text.model import Seq2SeqSql2TextModel
from darelabdb.nlp_text_to_sql.model import Seq2SeqText2SqlModel
from darelabdb.nlp_data_processing.processing_steps.processing_step_picker import (
    processing_step_picker,
)
from darelabdb.nlp_data_processing.sequential_processing_executor import (
    SequentialProcessingExecutor,
)
from darelabdb.utils_datasets.dataset_class_picker import dataset_class_picker
from darelabdb.nlp_models.model_class_picker import model_class_picker
from darelabdb.nlp_models.peft_class_picker import peft_class_picker
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

MODEL_TYPE = Union[Seq2SeqText2SqlModel, Seq2SeqSql2TextModel]


def _initialize_data_processors(
    config: dict,
) -> Tuple[SequentialProcessingExecutor, SequentialProcessingExecutor]:
    """
    Creates the instances of the preprocessor and postprocessor based on the given configuration.

    Args:
        config (dict): A dictionary with the values of the parameters for the initialization of the model.

    Returns (SequentialProcessingExecutor, SequentialProcessingExecutor): An instance of the preprocessor and
        postprocessor.

    """

    logger.info("Initializing the processors...")

    # Initialize preprocessor
    preprocessor = SequentialProcessingExecutor(
        input_steps=[
            (processing_step_picker(step["name"]), step["args"])
            for step in config["preprocessor"]["input_steps"]
        ],
        output_steps=(
            [
                (processing_step_picker(step["name"]), step["args"])
                for step in config["preprocessor"]["output_steps"]
            ]
            if config["preprocessor"]["output_steps"] is not None
            else None
        ),
    )

    # Initialize postprocessor
    if "postprocessor" in config:
        postprocessor = SequentialProcessingExecutor(
            input_steps=[
                (processing_step_picker(step["name"]), step["args"])
                for step in config["postprocessor"]["input_steps"]
            ],
        )
    else:
        postprocessor = None

    return preprocessor, postprocessor


def _initialize_model(
    config: dict, postprocessor: SequentialProcessingExecutor = None
) -> MODEL_TYPE:
    """
    Returns an instance of a model and a tokenizer based on the given configuration parameters.

    Args:
        config (dict): A dictionary with the values of the parameters for the initialization of the model.
        postprocessor (SequentialProcessingExecutor): The postprocessing pipeline that will be applied in the
            predictions of the model.

    Returns (MODEL_TYPE, PreTrainedTokenizerBase): MODEL_TYPE, PreTrainedTokenizerBase

    """

    logger.info("Initializing the model...")

    # Load arguments from config dict
    load_from_checkpoint = config["load_from_checkpoint"]
    model_class_name = config["model_class"]
    model_name_or_path = config["args"]["model_name_or_path"]
    model_args = config["args"]

    # Get the class of the model
    model_class = model_class_picker(model_class_name)

    # Get the PEFT config if there is one
    if "peft_config" in config:
        peft_config_class_name = config["peft_config"]["peft_config_class"]
        peft_config_args = config["peft_config"]["args"]
        # Create PEFT configuration
        peft_config_class = peft_class_picker(peft_config_class_name)
        peft_config = peft_config_class(peft_config_args)
    else:
        peft_config = None

    # Load the model
    if load_from_checkpoint:
        model = model_class.load_from_checkpoint(
            checkpoint_path=model_name_or_path,
            # NOTE: Without the line bellow, we get OOM when trying to load Llama
            map_location="cpu",
        )
    else:
        model = model_class(
            **model_args,
            postprocessor=postprocessor,
            peft_config=peft_config,
        )

    return model


def _initialize_data_module(
    data_module_config: dict,
    tokenizer: PreTrainedTokenizerBase,
    preprocessor: SequentialProcessingExecutor,
    decoder_only: bool,
) -> QueryDataModule:
    """
    Creates a QueryDataModule class based on the given configuration.

    Args:
        config (dict): A dictionary with the values of the parameters for the
            initialization of the model.
        tokenizer (PreTrainedTokenizerBase): The tokenizer that will be used in
            the data module.
        preprocessor (SequentialProcessingExecutor): The preprocessing pipeline
            that will be applied in the dataset.
        decoder_only (bool): Whether the data should be prepared for a decoder-only
            model or not

    Returns (QueryDataModule): A dataset in the format of a QueryDataModule.
    """

    logger.info("Initializing the data module...")

    # Load arguments from config dict
    dataset_name = data_module_config["dataset"]["name"]
    dataset_args = data_module_config["dataset"]["args"]
    data_module_args = data_module_config["args"]

    # Create dataset
    dataset_class = dataset_class_picker(dataset_name)
    dataset = dataset_class(**dataset_args)

    # Create data module
    data_module = to_data_module(
        dataset,
        preprocessor,
        tokenizer=tokenizer,
        decoder_only=decoder_only,
        **data_module_args,
    )

    return data_module


def factory(config: dict) -> Tuple[MODEL_TYPE, QueryDataModule]:
    """
    Creates a model and a data module based on the configuration parameters in the given configuration.

    Args:
        config (dict): A dictionary with the values of the parameters required for the creation of the model and
            the data module.

    Returns (MODEL_TYPE, QueryDataModule): (MODEL_TYPE, QueryDataModule)

    """

    # Initialize the processors
    preprocessor, postprocessor = _initialize_data_processors(config["processors"])

    # Initialize text-to-sql model
    model = _initialize_model(config["model"], postprocessor=postprocessor)

    # Initialize the data module
    data_module = _initialize_data_module(
        config["data_module"],
        tokenizer=model.tokenizer,
        preprocessor=preprocessor,
        decoder_only=model.decoder_only,
    )

    return model, data_module
