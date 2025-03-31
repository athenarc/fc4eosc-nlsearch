from typing import Callable

from darelabdb.nlp_data_processing.processing_steps import (
    add_sql_asc_order_explicitly,
    add_sql_whitespaces,
    text_assemble,
    lowercase_sql,
    remove_db_id,
    serialize_schema,
    correct_attribute_names,
    sql_remove_table_aliases,
    get_value_links,
    sql_transpile,
    remove_extra_whitespace,
    sql_keep_executable,
)

SUPPORTED_PROCESSING_STEPS = {
    "add_sql_asc_order_explicitly": add_sql_asc_order_explicitly,
    "add_sql_whitespaces": add_sql_whitespaces,
    "text_assemble": text_assemble,
    "lowercase_sql": lowercase_sql,
    "remove_db_id": remove_db_id,
    "serialize_schema": serialize_schema,
    "correct_attribute_names": correct_attribute_names,
    "sql_remove_table_aliases": sql_remove_table_aliases,
    "get_value_links": get_value_links,
    "sql_transpile": sql_transpile,
    "remove_extra_whitespace": remove_extra_whitespace,
    "sql_keep_executable": sql_keep_executable,
}


def processing_step_picker(processing_step_name: str) -> Callable:
    """
    Returns a processing step callable.

    Args:
        processing_step_name (str): The name of the requested processing step.

    Returns (Callable): Callable

    """
    if processing_step_name not in SUPPORTED_PROCESSING_STEPS:
        raise NotImplementedError(
            f"The processing step {processing_step_name} is not implemented. The supported processing steps "
            f"are: {list(SUPPORTED_PROCESSING_STEPS.keys())}"
        )

    return SUPPORTED_PROCESSING_STEPS[processing_step_name]
