from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql.utils.sql import sql_is_valid


def sql_keep_executable(
    datapoint: SqlQueryDatapoint,
    column_names_field: str = "column_names",
    table_names_field: str = "table_names",
    dialect: str = "mysql",
) -> SqlQueryDatapoint:
    executable = []

    for sql_query in datapoint.candidates:
        if sql_is_valid(
            sql_query,
            datapoint.db_schema,
            table_names_field,
            column_names_field,
            dialect=dialect,
        ):
            executable.append(sql_query)

    if len(executable) > 0:
        datapoint.prediction = executable[0]
    else:
        datapoint.prediction = None

    return datapoint
