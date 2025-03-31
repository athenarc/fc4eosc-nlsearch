from typing import List, Dict, Optional


def filter_schema(
    schema: List,
    keep_values: Optional[Dict] = None,
    exclude_values: Optional[Dict] = None,
) -> List:
    """
    Filters a database schema based on specified columns to keep or exclude, ensuring that foreign key references
    are updated accordingly. The function raises an error if non-existent tables or columns are specified in
    keep_values or exclude_values.

    Args:
        schema (List[Dict]): A list representing the database schema. Each item is a dictionary with a table name and
                             a list of columns. Each column includes attributes : column name, data type,
                             primary key status, and foreign keys.
        keep_values (Optional[Dict]): A dictionary specifying the tables and columns to retain in the output.
                                      Format example:
                                      {
                                          "table1": ["col1", "col2"],
                                          "table2": ["*"]  # Retain all columns for table2
                                      }
        exclude_values (Optional[Dict]): A dictionary specifying the tables and columns to exclude from the output.
                                         Format example:
                                         {
                                             "table1": ["col2", "col3"],
                                             "table2": ["*"]  # Exclude all columns and the table
                                         }

    Returns:
        List[Dict]: A filtered schema with the specified columns kept or excluded. If a table has no columns remaining
                    after filtering, it is excluded from the result. Returns an empty list if all tables are excluded.

    Raises:
        ValueError: If neither keep_values nor exclude_values is provided, or if both are provided simultaneously.
        ValueError: If a non-existent table or column is specified in keep_values or exclude_values.
    """

    # Input validation
    if keep_values is None and exclude_values is None:
        raise ValueError("One of keep_values or exclude_values must be provided")
    if keep_values and exclude_values:
        raise ValueError(
            "One of keep_values or exclude_values must be provided, but not both."
        )

    # Create a dictionary to track existing tables and their columns in the schema
    schema_map = {
        table["table_name"]: {col["column"] for col in table["columns"]}
        for table in schema
    }

    # Check for non-existent tables and columns in keep_values and exclude_values
    values_to_check = keep_values if keep_values else exclude_values
    for table_name, columns in values_to_check.items():
        if table_name not in schema_map:
            raise ValueError(f"Table '{table_name}' does not exist in the schema.")
        if columns != ["*"]:  # Skip wildcard case
            for column in columns:
                if column not in schema_map[table_name]:
                    raise ValueError(
                        f"Column '{column}' does not exist in table '{table_name}'."
                    )

    # Function to check if a column should be kept
    def should_keep(table_name, column):
        if keep_values:
            if table_name in keep_values:
                return (
                    keep_values[table_name] == ["*"]
                    or column in keep_values[table_name]
                )
            return False
        elif exclude_values:
            if table_name in exclude_values:
                return not (
                    exclude_values[table_name] == ["*"]
                    or column in exclude_values[table_name]
                )
            return True
        return True  # Default case, actually wont be used ever

    # Process the schema
    filtered_schema = []
    for table in schema:
        table_name = table["table_name"]
        columns = table["columns"]
        filtered_columns = []

        # Filter columns based on keep/exclude rules
        for col in columns:
            column_name = col["column"]

            if should_keep(table_name, column_name):
                # Remove any foreign keys that reference excluded columns
                if "foreign_keys" in col:
                    col["foreign_keys"] = [
                        fk
                        for fk in col["foreign_keys"]
                        if should_keep(fk["foreign_table"], fk["foreign_column"])
                    ]
                filtered_columns.append(col)

        # If exclude_values is used and all columns of a table are excluded, skip the entire table
        if exclude_values and not filtered_columns:
            continue  # Skip this table entirely

        # Add the table to the filtered schema if it has remaining columns
        if filtered_columns:
            filtered_schema.append(
                {"table_name": table_name, "columns": filtered_columns}
            )

    return filtered_schema
