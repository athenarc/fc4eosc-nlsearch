import json
import sqlite3


class DbSchema:
    """Stores information about the schema of a relational database"""

    table_names: list[str]
    column_names: list[tuple[str, str]]  # tuple(table_index, column_name)
    column_types: list[str]  # list with the type of each column
    foreign_primary_keys: list[
        tuple[str, str]
    ]  # tuple(<foreign_key_column_index>, <primary_key_column_index>)

    def __init__(self, db_path_or_json: str):
        """
        Initializes the information of a database.
        Args:
            db_path_or_json (str): The path to an .sqlite file or to a .json file containing the tables names, column
                                   names, column types and the foreign primary key relations of the database.
        """
        # If a .sqlite file has been given to get the database schema
        if db_path_or_json.endswith(".sqlite"):
            (
                self.table_names,
                self.column_names,
                self.column_types,
                self.foreign_primary_keys,
            ) = self._get_sqlite_schema(db_path_or_json)
        elif db_path_or_json.endswith(".json"):
            with open(db_path_or_json, "r") as f:
                schema_info = json.load(f)
                self.table_names = (
                    schema_info["tables_names_original"]
                    if "tables_names_original" in schema_info
                    else schema_info["table_names"]
                )
                self.column_names = (
                    schema_info["column_names_original"]
                    if "column_names_original" in schema_info
                    else schema_info["column_names"]
                )
                self.column_types = schema_info["column_types"]
                self.foreign_primary_keys = schema_info["foreign_keys"]

    def to_dict(self) -> dict:
        return {
            "table_names": self.table_names,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "foreign_primary_keys": self.foreign_primary_keys,
        }

    @staticmethod
    def _get_sqlite_schema(sqlite_path: str) -> tuple:
        """
        Returns the database schema of a .sqlite file.
        Args:
        sqlite_path (str): database path

        Returns (tuple): Returns a tuple with
                        'table_names': list[str],
                        'column_names': list[tuple(<table_index>, <column_name>)],
                        'column_types': list[str],
                        'foreign_primary_keys': list[tuple(<foreign_key_column_index, primary_key_column_index>)]
        """

        conn = sqlite3.connect(sqlite_path)
        conn.execute("pragma foreign_keys=ON")

        # Get all the tables
        table_names = list(
            map(
                lambda row: row[0],
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall(),
            )
        )

        # Initialize list of columns
        column_names = [(-1, "*")]
        column_types = []

        foreign_primary_keys_names = []
        # For each table
        for t_i, table_name in enumerate(table_names):
            # Get the foreign keys of the table
            table_fks_info = conn.execute(
                "PRAGMA foreign_key_list('{}') ".format(table_name)
            ).fetchall()
            for _, _, pk_table, fk_column, pk_column, _, _, _ in table_fks_info:
                foreign_primary_keys_names.append(
                    ((table_name, fk_column, pk_table, pk_column))
                )

            # Get the table's columns names and types
            table_columns_info = list(
                map(
                    lambda row: (row[1], row[2]),
                    conn.execute(
                        "PRAGMA table_info('{}') ".format(table_name)
                    ).fetchall(),
                )
            )
            # For every column
            for column_name, column_type in table_columns_info:
                column_names.append((t_i, column_name))
                column_types.append(column_type)

        # Convert foreign-primary keys names to column indexes
        foreign_primary_keys = []
        for fk_table, fk_column, pk_table, pk_column in foreign_primary_keys_names:
            foreign_primary_keys.append(
                (
                    column_names.index((table_names.index(fk_table), fk_column)),
                    column_names.index((table_names.index(pk_table), pk_column)),
                )
            )

        return table_names, column_names, column_types, foreign_primary_keys

    def get_schema_elements(self):
        """
        Returns the database schema as a dict of tables and its columns.
        """
        schema_elements = {}
        for t_i, table in enumerate(self.table_names):
            # Get the columns of the current table
            table_columns = list(
                filter(lambda column: column[0] == t_i, self.column_names)
            )
            # Get the columns names and assign them in the table
            schema_elements[table] = list(map(lambda column: column[1], table_columns))

        return schema_elements
