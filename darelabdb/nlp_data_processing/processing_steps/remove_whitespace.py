from typing import Optional
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def remove_extra_whitespace(
    datapoint: SqlQueryDatapoint, source_field: str, dest_field: Optional[str]
) -> SqlQueryDatapoint:
    """
    Removes extra whitespaces from a given field of a datapoint. The result will
    not have any trailing or leading whitespace and will only contain single
    spaces (i.e., no double spaces, newline characters, etc.).

    Args:
        datapoint (SqlQueryDatapoint): The datapoint that contains a field
            where the extra whitespace will be removed.
        source_field (str): The field of the datapoint where extra whitespace
            should be removed.
        dest_field (Optional[str]): The field of the datapoint where the result
            should be saved. If not given, then the result will be stored in the
            source field.

    Returns:
        SqlQueryDatapoint: The updated datapoint.
    """

    original_text = getattr(datapoint, source_field)

    # Remove trailing and leading whitespace
    processed_text = original_text.strip()

    # Remove extra whitespace characters inside the string
    processed_text = " ".join(processed_text.split())

    # Store the processed text in the datapoint
    if dest_field:
        setattr(datapoint, dest_field, processed_text)
    else:
        setattr(datapoint, source_field, processed_text)

    return datapoint
