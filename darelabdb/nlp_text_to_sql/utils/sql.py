from typing import Dict

import pglast
from sqlglot import exp, parse_one


def get_attributes(select_expression: exp.Select):
    """Identify table and column names used in the given SELECT expression"""
    # Get the tables used in the expression and their aliases which might be useful later
    used_tables = []
    table_aliases = {}
    for table in select_expression.find_all(exp.Table):
        used_tables.append(table.name)
        if table.alias:
            # If the table has an alias, add it for future reference
            table_aliases[table.alias] = table.name
    used_tables = list(set(used_tables))  # Avoid duplicates

    used_columns = []
    for column in select_expression.find_all(exp.Column):
        if column.name == "*":
            # No need to check star operators
            continue
        elif len(used_tables) == 1:
            # If there is only one table then use that table
            used_columns.append(f"{used_tables[0]}.{column.name}")
        elif column.table in used_tables:
            # Check that the table of the column is a used table
            used_columns.append(f"{column.table}.{column.name}")
        elif column.table in table_aliases:
            # Check that the table of the column is a table alias and replace it
            used_columns.append(f"{table_aliases[column.table]}.{column.name}")
        else:
            # If there are more than one tables but no table is specified for
            #   this column, then add it as it is and check it later
            used_columns.append(column.name)

    return used_tables, used_columns


def sql_is_valid(
    sql_query: str,
    db_schema: Dict,
    table_names_field: str,
    column_names_field: str,
    dialect: str = "mysql",
) -> bool:
    # Check syntax with pglast, for postgres dialect
    if dialect == "postgres":
        try:
            # Try to parse the given SQL query with pglast
            pglast.parse_sql(sql_query)
        except Exception as e:
            # If an exception is thrown, there is a syntax error
            return False

    # sqlglot can help us check the table and column names
    # NOTE: All name comparisons are case-insensitive using .casefold()
    try:
        # NOTE: The MySQL dialect is used because Spider doesn't differentiate single/double quotes
        parse = parse_one(sql_query, dialect=dialect)
    except Exception as e:
        # If an exception is thrown, there is a syntax error
        return False

    used_tables = []
    used_columns = []
    # Check all select statements separately to avoid errors due to nesting
    select_expressions = list(parse.find_all(exp.Select, bfs=False))
    for select_expression in reversed(select_expressions):
        # Go through expression from deepest to shallowest and pop them from the
        # AST to avoid parsing tables/columns more than once
        select_expression.pop()
        # Get tables and columns for this expression
        cur_used_tables, cur_used_columns = get_attributes(select_expression)
        # Add to global used tables and columns
        used_tables.extend(cur_used_tables)
        used_columns.extend(cur_used_columns)

    # Avoid duplicates
    used_tables = list(set(used_tables))
    used_columns = list(set(used_columns))

    # Load table and column names from schema
    db_tables = [table_name.casefold() for table_name in db_schema[table_names_field]]
    db_columns = [
        f"{db_tables[table_idx]}.{column_name.casefold()}"
        for table_idx, column_name in db_schema[column_names_field]
    ]

    # Check table names
    for used_table in used_tables:
        if used_table.casefold() not in db_tables:
            return False

    # Check column names
    for used_column in used_columns:
        if "." in used_column:
            # If we have the table specified in the column name
            if used_column.casefold() not in db_columns:
                return False
        else:
            # Otherwise we must search if this column exists in any of the used tables
            if not any(
                [
                    f"{used_table}.{used_column}".casefold() in db_columns
                    for used_table in used_tables
                ]
            ):
                return False

    # TODO: Check foreign keys

    # If none of the previous checks failed then the SQL query is valid
    return True
