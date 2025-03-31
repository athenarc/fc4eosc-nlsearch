from typing import Dict

import pandas as pd
from dotenv import load_dotenv

import wandb

load_dotenv(override=True)


def wandb_init(
    experiment_name: str, group_name: str, project_name: str, config: Dict
) -> None:
    wandb.init(
        project=project_name,
        name=experiment_name,
        group=group_name,
        config=config,
        entity="darelab",
        tags=[config["model"]["name"]],
    )


def wandb_log_results(
    results: pd.DataFrame, statistics: dict, experiment_time: int
) -> None:
    statistics["experiment_time_secs"] = experiment_time
    wandb.log(statistics)
    wandb.log({"results": wandb.Table(dataframe=results)})
    wandb.finish()
