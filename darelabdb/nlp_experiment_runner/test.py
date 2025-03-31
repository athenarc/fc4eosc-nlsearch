from typing import Dict, Optional
import argparse
import itertools
import json

import lightning as L
from darelabdb.nlp_experiment_runner.factory import factory
from darelabdb.nlp_experiment_runner.read_configuration import read_configuration


def test(config: Dict, predictions_out: Optional[str] = None):
    """
    Train a model based on the parameters given in the configuration dictionary.
    Args:
        config (Dict): The configuration dictionary.
        predictions_out (Optional[str]): Optional path to write predictions to.
    """

    model, data_module = factory(config)

    training_parameters = config["training_parameters"]

    trainer = L.Trainer(
        accelerator=training_parameters["accelerator"],
        devices=training_parameters["devices"],
        precision=training_parameters["precision"],
        strategy=training_parameters["strategy"],
        logger=False,
    )

    datapoints = trainer.predict(model, data_module)
    # Trainer.predict() returns a list of lists (where each inner list
    #   contains the datapoints of a batch) so it has to be flattened
    datapoints = list(itertools.chain.from_iterable(datapoints))

    if predictions_out:
        # Convert datapoints to a list of json object for writting the to a file
        datapoints_dump = [
            datapoint.model_dump(mode="json") for datapoint in datapoints
        ]
        # Write datapoints to file
        with open(predictions_out, "w") as fp:
            json.dump(datapoints_dump, fp)

    # Calculate metrics
    metric_scores = model.get_scores(datapoints)

    # Print scores
    print(f"{10*'='} Scores {10*'='}")
    for metric_name, score in metric_scores.items():
        print(f"{metric_name}:\t{score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a model from a configuration file"
    )
    parser.add_argument(
        "--config_file", help="path to config file", type=str, required=True
    )
    parser.add_argument(
        "--predictions_out",
        help="The path to a file were predictions should be saved (in json format)",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    config = read_configuration(args.config_file)

    test(config, args.predictions_out)
