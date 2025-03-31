import re
from typing import Literal

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def text_assemble(
    datapoint: SqlQueryDatapoint,
    recipe: str,
    dest_field: Literal["model_input", "expected_output"],
) -> SqlQueryDatapoint:
    """
    Assembles a textual field based on the given recipe and saves it under the dest_field field.
        The recipe should be a string with references to datapoint fields in the following format: `{{datapoint.field_name}}`.
        The double brackets and everything inside them will be replaced by the value stored in `field_name`.
        For example, if we want to combine the fields db_id and sql_query with a '|' to separate them, the recipe would
        be `{{datapoint.db_id}} | {{datapoint.sql_query}}`.

    Args:
        datapoint (SqlQueryDatapoint): The datapoint to perform the processing step on.
        recipe (str): The recipe for creating the textual field.
        dest_field (Literal["model_input", "expected_output"]): The field where the assembled text should be saved.

    Returns:
        SqlQueryDatapoint: The given datapoint with the dest_field updated.
    """

    result = recipe

    for match in re.finditer(r"({{datapoint.(\w+)}})", recipe):
        reference, field_name = match.groups()
        result = result.replace(reference, getattr(datapoint, field_name), 1)

    setattr(datapoint, dest_field, result)

    return datapoint


def copy_field(
    datapoint: SqlQueryDatapoint,
    origin_field: str,
    dest_field: str,
) -> SqlQueryDatapoint:
    """
    Sets the value of a field by copying an existing field of the datapoint.

    Args:
        datapoint (SqlQueryDatapoint): The datapoint to perform the processing step on.
        origin_field (str): The origin field to copy the value from.
        dest_field (str): The destination field where the value will be copied.

    Returns:
        SqlQueryDatapoint: The given datapoint with the dest_field updated.
    """

    value = getattr(datapoint, origin_field)
    setattr(datapoint, dest_field, value)

    return datapoint
