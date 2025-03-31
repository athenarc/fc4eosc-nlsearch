import os
from datetime import datetime

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
from loguru import logger


def initialise_checkpoint_callback(
    checkpoint_args: dict, experiment_name: str
) -> ModelCheckpoint:
    """
    Initialises the checkpoint callback of a 'lighting.Trainer'.
    Args:
        checkpoint_args (dict): A dictionary with the arguments of the Callback, plus the 'storage_path' argument.
        experiment_name (str):  The name of the trainer's experiment. It is used combined with the 'storage path' in the
            'checkpoint_args' to create the 'dirpath' argument.

    Returns: 'lightning.pytorch.callbacks.ModelCheckpoint'

    """
    if "dirpath" in checkpoint_args:
        checkpoints_dir_path = checkpoint_args["dirpath"]
    else:
        checkpoints_dir_path = (
            f"components/darelabdb/nlp_nl_search/storage/checkpoints/{experiment_name}"
        )

    if os.path.exists(checkpoints_dir_path):
        logger.info(
            f"There are existing checkpoints in the {checkpoints_dir_path}! "
            f"The checkpoints dir path will be changed!"
        )
        checkpoints_dir_path = (
            f"{checkpoints_dir_path}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

    checkpoint_args["dirpath"] = checkpoints_dir_path

    logger.info(
        f"Checkpoint callback has been configured. The checkpoints will be saved in {checkpoints_dir_path}"
    )

    return ModelCheckpoint(**checkpoint_args)


def initialise_early_stopping_callback(early_stopping_args: dict) -> EarlyStopping:
    """
    Initialises the early stopping callback of a 'lighting.Trainer'.
    Args:
        early_stopping_args(dict): A dictionary with the arguments of the Callback.

    Returns: 'lightning.pytorch.callbacks.EarlyStopping'

    """

    early_stopping = EarlyStopping(**early_stopping_args)

    logger.info("Early stopping callback has been configured.")

    return early_stopping


def initialise_callbacks(config: dict) -> list[Callback]:
    """
    Initialises the callbacks of the 'lighting.Trainer'.
    Args:
        config (dict): The configuration parameters for the callbacks.

    Returns: list['lightning.pytorch.callbacks.callback.Callback']

    """
    callbacks = []

    # Initialize the checkpoint callback
    if "checkpoints" in config:
        callbacks.append(
            initialise_checkpoint_callback(
                checkpoint_args=config["checkpoints"],
                experiment_name=config["experiment_name"],
            )
        )
    else:
        logger.info(
            "Checkpoint callback was not configured. The default behavior of the trainer will be applied."
        )

    # Initialize the early stopping callback
    if "early_stopping" in config:
        callbacks.append(initialise_early_stopping_callback(config["early_stopping"]))
    else:
        logger.info(
            "Early stopping callback was not configured. The default behavior of the trainer will be applied."
        )

    return callbacks
