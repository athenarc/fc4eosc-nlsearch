from typing import Optional

import pandas as pd
from loguru import logger
from darelabdb.utils_query_analyzer import MoQueryExtractor


def add_queries_categories(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Adds in the given dataframe a column with the categories of the SQL queries existing in the dataframe.

    Args:
        dataset (pd.DataFrame): Dataframe containing the SQL queries in a column named 'SQL'.
    """

    sql_extractor = MoQueryExtractor()

    def get_sql_category(sql_query: str) -> Optional[str]:

        try:
            sql_query_info = sql_extractor.extract(sql_query)
        except Exception as e:
            logger.warning(
                f"There was an error in extracting the query {sql_query}. Its category will be set to None\n"
                f"Error: {e}"
            )
            return None

        return (
            sql_query_info.get_structural_category(return_format="name")
            + "-"
            + sql_query_info.get_operator_types_category(return_format="name")
        )

    dataset["SQL category"] = dataset["SQL"].apply(get_sql_category)

    return dataset
