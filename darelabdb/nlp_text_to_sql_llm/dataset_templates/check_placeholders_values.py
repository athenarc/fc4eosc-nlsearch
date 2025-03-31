import json
from tqdm import tqdm
import pandas as pd
from loguru import logger
from darelabdb.utils_database_connector.core import Database


def empty_results_of_filled_templates(
    dataset_templates: pd.DataFrame, placeholders_values: dict, db: Database
):
    """Checks that the values assigned in the placeholders do not result in empty query results"""

    empty_results = []
    for _, row in tqdm(dataset_templates.iterrows(), desc="Execute queries..."):
        sql_query = row["sql_query_without_values"].format(**placeholders_values)

        result = db.execute(sql_query)

        if type(result) is dict:
            logger.info(
                f"Query template {row['sql_query_without_values']} could not be executed. Error : {result['error']}"
            )
        elif result.empty:
            logger.info(
                f"Query template with empty results: {row['sql_query_without_values']}"
            )
            empty_results.append(
                {
                    "template": row["sql_query_without_values"],
                    "filled_template": sql_query,
                }
            )

    return empty_results


if __name__ == "__main__":

    query_templates = pd.read_excel(
        "development/faircore_nl_search/storage/faircore_final_benchmark.xlsx",
        sheet_name="Benchmark",
    )[["sql_query_without_values"]]

    with open(
        "development/faircore_nl_search/storage/query_templates_candidate_values.json"
    ) as f:
        placeholders_values_scenarios = json.load(f)

    db = Database("fc4eosc")

    for scenario, placeholder_values in placeholders_values_scenarios.items():

        empty_queries = empty_results_of_filled_templates(
            dataset_templates=query_templates.dropna(),
            placeholders_values=placeholder_values,
            db=db,
        )

        print(f"There are {len(empty_queries)} queries that have empty query results!")
