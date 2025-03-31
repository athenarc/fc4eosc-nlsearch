import argparse
import yaml
from typing import Optional

from darelabdb.nlp_experiment_runner.train import train


def sweep(
    config_file_path: str,
    learning_rate: Optional[float],
    optimizer: Optional[str],
    accumulate_grad_batches: Optional[int],
    lora_r: Optional[int],
):
    """
    Start training based on a configuration file, but first change some of the
    parameters in the configuration based on the given arguments. This function
    is useful for starting a runner for a WandB sweep.

    Args:
        config_file_path (str): The path to the base configuration file.
        learning_rate (float): The learning rate to use for this run.
    """
    # Read template config
    with open(config_file_path, "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # Set given arguments in the config
    # NOTE: This section can be extended to as many arguments we want
    if learning_rate:
        config["model"]["learning_rate"] = learning_rate
    if optimizer:
        config["model"]["optimizer"] = optimizer
    if accumulate_grad_batches:
        config["training_parameters"][
            "accumulate_grad_batches"
        ] = accumulate_grad_batches
    if lora_r:
        config["model"]["peft_config"]["args"]["r"] = lora_r

    # Start training from the updated config
    train(config)


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    args = parser.parse_args()

    sweep(
        args.config,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        accumulate_grad_batches=args.accumulate_grad_batches,
        lora_r=args.lora_r,
    )
