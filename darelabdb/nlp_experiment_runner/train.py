import os
from typing import Dict
from datetime import datetime
import argparse

import lightning as L
import wandb
from darelabdb.nlp_experiment_runner.callbacks import initialise_callbacks
from darelabdb.nlp_experiment_runner.factory import factory
from darelabdb.nlp_experiment_runner.read_configuration import read_configuration
from dotenv import load_dotenv
from lightning.pytorch.loggers import Logger, WandbLogger
from loguru import logger


def initialise_model_logger(logger_config: Dict, run_config: Dict) -> Logger:
    """
    Initialises the logger of a 'lighting.Trainer'.
    Args:
        logger_config (Dict): A dictionary with the logger 'name' and its 'args'.
        run_config (Dict): A dictionary with the configuration parameters that
            will be saved for the run.

    Returns: 'lightning.pytorch.loggers.Logger'
    """
    # Initialize the logger of the model
    if logger_config["name"] == "wandb":
        wandb_args = logger_config["args"]

        # Login to wandb
        wandb.login(key=os.getenv("wandb_api_key"))

        model_logger = WandbLogger(**wandb_args, config=run_config)

    else:
        raise NotImplementedError(
            f"The requested logger ({logger_config['name']}) is not supported!"
        )

    return model_logger


def train(config: Dict):
    """
    Train a model based on the parameters given in the configuration.
    Args:
        config_file (Dict): The configuration for training.

    Returns:

    """
    load_dotenv()

    model, data_module = factory(config)

    training_parameters = config["training_parameters"]

    experiment_name = training_parameters["experiment_name"]
    if experiment_name is None or experiment_name == "":
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(
            f"Experiment name was not provided. The experiment name: {experiment_name} was given to the "
            f"current experiment"
        )

    # Initialize the logger of the model
    if "logger" in training_parameters:
        model_logger = initialise_model_logger(
            logger_config=training_parameters["logger"],
            run_config=config,
        )
    else:
        model_logger = None
    logger.info(
        f"Logger have been initialized. loggers = {type(model_logger).__name__}"
    )

    callbacks = initialise_callbacks(training_parameters)

    trainer = L.Trainer(
        max_epochs=training_parameters["max_epochs"],
        log_every_n_steps=training_parameters["log_every_n_steps"],
        accelerator=training_parameters["accelerator"],
        devices=training_parameters["devices"],
        accumulate_grad_batches=training_parameters["accumulate_grad_batches"],
        precision=training_parameters["precision"],
        strategy=training_parameters["strategy"],
        logger=model_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model from a configuration file"
    )
    parser.add_argument(
        "--config_file", help="path to config file", type=str, required=True
    )
    args = parser.parse_args()

    config = read_configuration(args.config_file)

    train(config)
