import argparse

import yaml


def read_configuration(config_path=None) -> dict:
    """
    Reads the configuration file.
    Args:
        config_path (str): The path of the configuration file. If None, the configuration path is read from the input.

    Returns (dict): The contents of a .yaml file.

    """
    # Read config
    if config_path is None:
        parser = argparse.ArgumentParser(description="NL2SQL")
        parser.add_argument(
            "--config_file", help="path to config file", type=str, required=True
        )
        args = parser.parse_args()
        config_file = args.config_file
    else:
        config_file = config_path

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config
