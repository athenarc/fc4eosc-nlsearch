from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def serialize_schema(
    datapoint: SqlQueryDatapoint,
    with_fp_relations: bool = False,
    with_db_id: bool = False,
    with_value_links: bool = False,
    db_id_sep: str = " | ",
    table_sep: str = " | ",
    columns_open_char: str = " : ",
    columns_close_char: str = "",
    column_sep: str = " , ",
    fp_relations_sep: str = " | ",
    fp_relation_sep: str = " , ",
    value_sep: str = ", ",
    value_open_char: str = "(",
    value_close_char: str = ")",
    column_names_field: str = "column_names",
    table_names_field: str = "table_names",
) -> SqlQueryDatapoint:
    """
    Serializes the schema of the datapoint to a string.

    Args:
        datapoint (SqlQueryDatapoint):The Datapoint that contains a schema that will be serialized.
        with_fp_relations (bool): If True the serialization will contain the foreign-primary key relations of
                                  the schema
        with_db_id (bool): Whether to include the database id in the serialization serialized
        with_value_links (bool): Whether to include value links found in a previous processing step
        db_id_sep (str): The separator string between the db id and the rest of the schema
        table_sep (str): The separation string between two tables of the schema
        columns_open_char (str): The opening string before serializing the columns of a table
        columns_close_char (str): The closing string after serializing the columns of a table
        column_sep (str): The separation string between two columns of the schema
        fp_relations_sep (str): The separation string between the foreign-primary key relations and the rest of
                                the schema
        fp_relation_sep (str): The separation string between two foreign-primary key relations of the schema
        value_sep (str): The separation string between two values that were linked to a column
        value_open_char (str): The opening string before serializing linked values
        value_close_char (str): The closing string after serializing linked values
        column_names_field (str): The field in the db_schema where the column names are found
        table_names_field (str): The field in the db_schema where the table names are found

    Returns:
        SqlQueryDatapoint: The datapoint with an updated serialized_schema field and a model_input field that has the
                           SQL and serialized schema.
    """

    # Get the columns and the tables from the schema
    columns = datapoint.db_schema[column_names_field]
    table_names = datapoint.db_schema[table_names_field]

    serialized_tables = []
    # For each table
    for t_i, table_name in enumerate(table_names):
        # Get the columns of the current table
        table_columns = list(filter(lambda column: column[0] == t_i, columns))
        # Keep only the name of the columns
        table_columns_names = list(map(lambda column: column[1], table_columns))

        # Serialize the columns of the table
        if with_value_links and table_name in datapoint.value_links:
            # If we have discovered value links for a column in this table
            columns_with_links = [
                (
                    f"{column_name} {value_open_char}{value_sep.join(datapoint.value_links[table_name][column_name])}{value_close_char}"
                    if column_name in datapoint.value_links[table_name]
                    else column_name
                )
                for column_name in table_columns_names
            ]
            serialized_columns = column_sep.join(columns_with_links)

        else:
            serialized_columns = column_sep.join(table_columns_names)

        # Create the complete serialization of the table with its columns
        serialized_table = (
            f"{table_name}{columns_open_char}{serialized_columns}{columns_close_char}"
        )
        serialized_tables.append(serialized_table)

    serialized_schema = f"{table_sep}".join(serialized_tables)

    # If the serialization contains foreign-primary key relations
    if with_fp_relations:
        fp_pairs = datapoint.db_schema["foreign_keys"]
        if len(fp_pairs) > 0:
            serialized_fp_relations = []
            # For each foreign-primary key relation
            for foreign_key_id, primary_key_id in fp_pairs:
                # Get the foreign and primary key in a format: <table_name>.<column_name>
                foreign_key_name = f"{table_names[columns[foreign_key_id][0]]}.{columns[foreign_key_id][1]}"
                primary_key_name = f"{table_names[columns[primary_key_id][0]]}.{columns[primary_key_id][1]}"

                serialized_fp_relations.append(
                    f"{foreign_key_name} = {primary_key_name}"
                )

            # Append the fp-relations to the serialized schema
            serialized_schema += fp_relations_sep + f"{fp_relation_sep}".join(
                serialized_fp_relations
            )

    # If the db_id will be included in the schema serialization
    if with_db_id:
        serialized_schema = f"{datapoint.db_id}{db_id_sep}{serialized_schema}"

    # Save the serialized schema
    datapoint.serialized_schema = serialized_schema

    return datapoint
