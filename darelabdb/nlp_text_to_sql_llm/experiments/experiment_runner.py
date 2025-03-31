import os
import time
from darelabdb.utils_datasets.dataset_abc import Dataset
import yaml
from loguru import logger
from tqdm import tqdm
import pandas as pd
from typing import Optional

from darelabdb.nlp_embeddings.embedding_methods.embedding_method_picker import (
    pick_embedding_method,
)
from darelabdb.nlp_embeddings.embedding_storage.ChromaDB import ChromaDB
from darelabdb.nlp_metrics.execution_accuracy import ExecutionAccuracy
from darelabdb.nlp_metrics.partial_match.partial_match import PartialMatchMetric
from darelabdb.nlp_text_to_sql_llm.experiments.experiment_config_model import (
    ExperimentConfig,
)
from darelabdb.nlp_text_to_sql_llm.experiments.experiment_tracking import (
    wandb_init,
    wandb_log_results,
)
from darelabdb.nlp_text_to_sql_llm.ollama import handle_text_to_sql
from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_examples_selector_picker import (
    pick_similar_examples_selector,
)
from darelabdb.utils_datasets.dataset_class_picker import dataset_class_picker
from loguru import logger
from tqdm import tqdm


def _add_partial_match_metric(results: pd.DataFrame) -> None:
    logger.info(f"Calculating partial match...")
    partial_match = PartialMatchMetric()
    partial_match.update(
        preds=results["predicted_sql_query"].values.tolist(),
        targets=results["sql_query"].values.tolist(),
        db_paths=results["db_path"].values.tolist(),
    )
    partial_match_results = partial_match.compute(aggregated=False)

    results["structural_match"] = partial_match_results["SM"]
    results["operator_match"] = partial_match_results["OM"]
    results["variable_match"] = partial_match_results["VM"]


def _add_execution_accuracy_metric(results: pd.DataFrame) -> None:
    logger.info(f"Calculating execution accuracy...")
    exec_acc = ExecutionAccuracy()
    exec_acc.update(
        preds=results["predicted_sql_query"].values.tolist(),
        targets=results["sql_query"].values.tolist(),
        db_paths=results["db_path"].values.tolist(),
    )

    exec_results = exec_acc.compute(aggregated=False, report_only_exec=False)
    exec_results = pd.DataFrame.from_records(exec_results)
    for column in exec_results.columns:
        results[column] = exec_results[column]


def _calculate_total_statistics(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "AVG exec": results["exec"].mean(),
                "AVG exec-only_common_result_columns": results[
                    "exec-only_common_result_columns"
                ].mean(),
                "AVG exec-target_result_columns_subset": results[
                    "exec-target_result_columns_subset"
                ].mean(),
                "AVG exec-target_result_columns_superset": results[
                    "exec-target_result_columns_superset"
                ].mean(),
                "Total exec errors": results["exec_error"].count(),
                "AVG structural_match": results["structural_match"].mean(),
                "AVG operator_match": results["operator_match"].mean(),
                "AVG variable_match": results["variable_match"].mean(),
            }
        ]
    )


def calculate_metrics_and_statistics(results: pd.DataFrame):
    """Updates the given dataframe with metrics and calculates total statistics."""
    # Add the values of partial match
    _add_partial_match_metric(results)

    # Add the values of execution accuracy metric
    _add_execution_accuracy_metric(results)

    # CALCULATE TOTAL STATISTICS

    total_statistics = _calculate_total_statistics(results)

    return total_statistics


def _get_dataset(dataset_name: str, download_dir: Optional[str] = None):
    #ensure that download_dir ens with slash
    if download_dir :
        download_dir = download_dir if download_dir.endswith("/") else download_dir + "/"
    if "." in dataset_name:
        dataset_name, split = dataset_name.split(".")
        dataset_class = dataset_class_picker(dataset_name)
        # Only pass cache_dir to Dataset subclasses
        instance = dataset_class(cache_dir=download_dir) if issubclass(dataset_class, Dataset) else dataset_class()
        return instance.get()[split]
    else:
        dataset_class = dataset_class_picker(dataset_name)
        instance = dataset_class(cache_dir=download_dir) if issubclass(dataset_class, Dataset) else dataset_class()
        dataset = instance.get()
        if type(dataset) is dict:
            unified_dataset = []
            for split, split_dataset in dataset.items():
                unified_dataset.extend(split_dataset)
            return unified_dataset
        else:
            return dataset

def ollama_experiment_runner(config: ExperimentConfig) -> None:
    """
    Retrieves the SQL predictions and calculates evaluation metrics (e.g., execution
    accuracy) and statistics (e.g., execution time) for the given run configuration.

    Args:
        config: The configuration parameters of the experiment.
    """

    # Set up the logging for the experiment
    experiment_logging_config = config.experiment_logging
    if experiment_logging_config.enable_wandb:
        wandb_init(
            experiment_name=experiment_logging_config.experiment_name,
            group_name=experiment_logging_config.group_name,
            project_name=experiment_logging_config.project_name,
            config=config.model_dump(mode="json"),
        )

    # CREATE POOL OF SIMILAR EXAMPLES
    examples_retriever = None
    if config.prompt.similar_examples is not None:
        # Create an In-Memory database with the examples pool
        embedding_db_name = "_".join(
            config.prompt.similar_examples.examples_pool
        ).replace(".", "_")
        examples_db = ChromaDB(
            db_name=embedding_db_name,
            embedding_col_name="embedding",
            primary_key_col_name="nl_question",
        )

        text_embedding_method = pick_embedding_method(
            config.prompt.similar_examples.embedding_method
        )
        if not examples_db.is_populated():
            examples_pool = config.prompt.similar_examples.examples_pool
            if examples_pool is None or len(examples_pool) == 0:
                raise ValueError(
                    "There must be at least one dataset in the similar examples pool!"
                )

            examples_dataset = []
            for dataset_name in examples_pool:
                examples_dataset.extend(_get_dataset(dataset_name, download_dir=config.download_dir))

            # Keep only the required information from the datasets and create a dict with the values of each column
            examples = {
                "nl_question": [],
                "sql_query": [],
                "db_id": [],
                "db_schema": [],
                "embedding": [],
            }

            for datapoint in tqdm(examples_dataset, desc="Creating demonstration pool"):
                examples["nl_question"].append(datapoint.nl_query)
                examples["embedding"].append(
                    text_embedding_method.get_item_embedding(datapoint.nl_query)
                )
                examples["sql_query"].append(datapoint.sql_query)
                examples["db_id"].append(datapoint.db_id)
                examples["db_schema"].append(datapoint.db_schema)

            examples_db.populate(rows=examples)

        examples_retriever = pick_similar_examples_selector(
            selector_name=config.prompt.similar_examples.selector,
            embedding_db=examples_db,
            text_embedding_method=text_embedding_method,
            # Skip the first example if the dataset of the similar examples' corpus contains the evaluated dataset
            skip_n=(
                1
                if config.evaluated_dataset
                in config.prompt.similar_examples.examples_pool
                else 0
            ),
        )

    # GET THE DATASET FOR THE EVALUATION
    dataset = _get_dataset(config.evaluated_dataset, download_dir=config.download_dir)
    start = time.time()

    # GET PREDICTIONS
    results = []
    for datapoint in tqdm(dataset, desc="Get model predictions..."):
        handle_text_to_sql(
            datapoint=datapoint,
            config={
                "model": config.model.model_dump(mode="json"),
                "prompt": config.prompt.model_dump(mode="json"),
            },
            example_retriever=examples_retriever,
        )
        results.append(
            {
                "nl_question": datapoint.nl_query,
                "sql_query": datapoint.sql_query,
                "db_path": datapoint.db_path,
                "predicted_sql_query": datapoint.prediction,
                "model_input": datapoint.model_input,
                "generated_text": datapoint.model_output,
            }
        )
    results = pd.DataFrame(results)

    # CALCULATE METRICS

    # Add the values of partial match
    _add_partial_match_metric(results)

    # Add the values of execution accuracy metric
    _add_execution_accuracy_metric(results)

    # CALCULATE TOTAL STATISTICS

    total_statistics = _calculate_total_statistics(results)

    # SAVE RESULTS

    if experiment_logging_config.save_local:
        save_dir = experiment_logging_config.save_dir
        if experiment_logging_config.save_dir is None:
            logger.warning(
                "No save directory provided! The results will be saved in the current directory"
            )
            save_dir = os.getcwd()

        # Store the results in a xlsx
        res_file_name = f"{save_dir}/{experiment_logging_config.experiment_name}.xlsx"
        with pd.ExcelWriter(res_file_name, engine="openpyxl") as writer:
            pd.json_normalize(config.model_dump(mode="json")).to_excel(
                writer, sheet_name="Model config"
            )
            results.to_excel(writer, sheet_name="Predictions", engine="xlsxwriter")
            total_statistics.to_excel(writer, sheet_name="Total statistics")

    if experiment_logging_config.enable_wandb:
        wandb_log_results(
            results,
            total_statistics.to_dict("records")[0],
            experiment_time=int(time.time() - start),
        )

    logger.info(f">>> Experiment finished in {time.time() - start} seconds.")
