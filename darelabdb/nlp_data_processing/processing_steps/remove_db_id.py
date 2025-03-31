from loguru import logger
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def _remove_db_id(sql_query: str, separator: str) -> str:
    if f" {separator} " not in sql_query:
        logger.warning(f"Separator '{separator}' not found in '{sql_query}'")
        return None

    _, new_sql_query = sql_query.split(sep=f" {separator} ", maxsplit=1)

    return new_sql_query


def remove_db_id(
    data_point: SqlQueryDatapoint, separator: str = "|", expect_candidates: bool = False
) -> SqlQueryDatapoint:
    """
    Removes a leading db_id and separator from the prediction of a text-to-SQL model.
    For example the raw prediction "singer | SELECT * FROM concert", will be
    processed to "SELECT * FROM concert".

    Args:
        data_point (SqlQueryDatapoint): The Datapoint that contains a prediction
            that we remove the db_id from.
        separator (str, optional): The separator that is expected between the
            db_id and the SQL prediction. Defaults to '|'.
        expect_candidates (bool): Whether to expect multiple candidates, instead
            of a single prediction. Deafults to False.

    Returns:
        SqlQueryDatapoint: The given Datapoint with the db_id removed from the prediction field.
    """
    if expect_candidates:
        new_candidates = [
            _remove_db_id(candidate, separator) for candidate in data_point.candidates
        ]
        data_point.candidates = new_candidates
    else:
        data_point.prediction = _remove_db_id(data_point.prediction, separator)

    return data_point
