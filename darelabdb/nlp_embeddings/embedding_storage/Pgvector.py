import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from darelabdb.nlp_embeddings.embedding_storage.EmbeddingDB import EmbeddingDB
from dotenv import load_dotenv
from loguru import logger
from psycopg2.extras import execute_values


class PgVector(EmbeddingDB):
    def __init__(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str,
        schema_name: str,
        table_name: str,
        column_types: dict,
        embedding_col_name: str,
        primary_key_col_name: str = None,
        index_name: str = "embeddings_index",
    ):
        """
        Initialize the EmbeddingDB object.

        Args:
            host: The host of the database.
            port: The port of the database.
            user: The user of the database.
            password: The password of the database.
            db_name: The name of the database.
            schema_name: The name of the schema that will be used to store the embeddings.
            table_name: The name of the table that will be used to store the embeddings.
            column_types: A dictionary with the names of the columns and their values. The types of the columns must be
                types supported in Postgres. To define an embedding column it must be of type
                `vector(<embedding_dimension>)`.
            embedding_col_name: The name of the column that will be used to store the embeddings.
                When inserting the rows, this name should appear in the rows dictionary.
            primary_key_col_name: (Optional) The name of the column that will be used as the primary key.
                When inserting the rows, this name should appear in the rows dictionary.
            index_name: The name of the embedding index.
        """
        super().__init__(db_name, embedding_col_name, primary_key_col_name)

        self.schema_name = schema_name
        self.table_name = table_name
        self.column_types = column_types
        self.index_name = index_name

        if (
            self.primary_key_col_name is not None
            and self.primary_key_col_name not in self.column_types
        ):
            raise ValueError(
                f"The column {self.primary_key_col_name} does not exist in the column_types dictionary."
            )

        if embedding_col_name not in self.column_types:
            raise ValueError(
                "The column with the embeddings must be in the column_types."
            )

        if not re.fullmatch(
            r"^vector\((\d+)\)$", self.column_types[self.embedding_col_name]
        ):
            raise ValueError(
                "The column with the embeddings must be of type vector(<embedding_dimension>)."
            )

        # Temporary solution until pgvector is added to our database
        conn_info = {
            "dbname": self.db_name,
            "user": user,
            "host": host,
            "password": password,
            "port": port,
        }

        self.conn_info = conn_info

    def initialize(self) -> None:
        """
        Initialize the schema and table that will be used to store the embeddings.
        """
        conn = psycopg2.connect(**self.conn_info)
        logger.info(
            "Initialising schema and table for the embeddings if they do not exist..."
        )

        column_creation = ""
        for column_name, column_type in self.column_types.items():
            if column_name == self.primary_key_col_name:
                column_creation += f"{column_name} {column_type} PRIMARY KEY, "
            else:
                column_creation += f"{column_name} {column_type}, "
        column_creation = column_creation[:-2]  # Remove the last comma and space

        with conn:
            cur = conn.cursor()
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS
                    {self.schema_name}.{self.table_name}
                    ({column_creation});
                    """
            )

    def initialize_index(self) -> None:
        """
        Initialize the index that will be used to speed up the similarity search.
        Ideally you could run this after the embeddings have been inserted.
        """
        conn = psycopg2.connect(**self.conn_info)

        logger.info("Initialising index for the embeddings if it does not exist...")
        with conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.index_name}
                ON {self.schema_name}.{self.table_name}
                USING hnsw ({self.embedding_col_name} vector_cosine_ops);
                """
            )

    def populate(self, rows: Dict, batch_size: int = 10_000) -> None:
        """
        Insert multiple item embeddings into the database.

        Args:
            rows: A dictionary where the keys are the column names and the values are the values of the
                columns. e.g.,
                {
                    "value_col": ["a", "b", "c"],
                    "embedding_col": [[0.1, 0.2], [0.3, 0.1], [0.6, 0.4]]
                }
            batch_size: The number of embeddings that will be inserted in each batch.
        """
        if self.embedding_col_name not in rows:
            raise ValueError(
                f"The column {self.embedding_col_name} does not exist in the item_embeddings dictionary."
            )

        if (
            self.primary_key_col_name is not None
            and self.primary_key_col_name not in rows
        ):
            raise ValueError(
                f"The column {self.primary_key_col_name} does not exist in the item_embeddings dictionary."
            )

        # Transform the dictionary into a list of tuples (rows)
        inserted_rows = list(map(list, zip(*list(rows.values()))))
        column_names = list(rows.keys())

        if set(column_names) != set(self.column_types.keys()):
            raise ValueError(f"Ech row must contain the columns {self.column_types}")

        self.initialize()

        def get_chunk():
            for i in range(0, len(inserted_rows), batch_size):
                # batch = [(item_id, embedding.tolist()) for item_id, embedding in item_embeddings[i:i + batch_size]]
                yield inserted_rows[i : i + batch_size]

        conn = psycopg2.connect(**self.conn_info)

        with conn:
            cur = conn.cursor()

            for ind, chunk in enumerate(get_chunk()):
                execute_values(
                    cur,
                    f"INSERT INTO {self.schema_name}.{self.table_name} ({', '.join(column_names)}) VALUES %s",
                    chunk,
                )
                logger.info(
                    f"Inserted embedding batch {ind + 1}/{(len(rows) // batch_size) + 1} "
                    f"of size {batch_size}"
                )

        self.initialize_index()

    def get_embedding(self, row_id: str) -> Optional[List[float]]:
        """
        Get the embedding of the given item.

        Args:
            row_id: The id (primary key) of the row that we want to get the embedding of.

        Returns:
            The embedding of the item or none if a primary key column was not specified upon initialization ot the
            given primary key was not found.
        """
        if self.primary_key_col_name is None:
            logger.error(
                "You cannot select an embedding without specifying a primary key column."
            )
            return None

        conn = psycopg2.connect(**self.conn_info)

        with conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT {self.embedding_col_name} FROM {self.schema_name}.{self.table_name} "
                f"WHERE {self.primary_key_col_name} = %s;",
                (row_id,),
            )
            res = cur.fetchone()

            return res[0] if res else None

    def get_neighbors(
        self,
        embedding: List[float],
        num: int,
        eq_filters: dict = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the neighbors of the given embedding using the cosine similarity operator of pgvector (<=>).
        Args:
            embedding: An embedding of an item that we want to find the neighbors of.
            num: The number of neighbors that we want to get.
            eq_filters: A dictionary with {<column_name>: <value>} to be applied, to exclude rows.
            columns: The columns that will be returned in the result.

        Returns:
            A pandas DataFrame with the columns specified in the columns parameter and the similarity score.
        """
        conn = psycopg2.connect(**self.conn_info)
        select_clause = ", ".join(columns) if columns is not None else "*"

        where_clause = ""
        if eq_filters is not None:
            where_clause = "WHERE "
            conditions = [f"{col} = {value}" for col, value in eq_filters.items()]
            where_clause += f"{' AND '.join(conditions)}"

        with conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT {select_clause}, 1 - (embedding <=> %s::vector) AS similarity
                FROM {self.schema_name}.{self.table_name}
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding, embedding, num),
            )

            res = cur.fetchall()
            # df = pd.DataFrame(res, columns=["id", "score"]).set_index("id")["score"]
            df = (
                pd.DataFrame(res, columns=columns + ["similarity"])
                if columns is not None
                else pd.DataFrame(res, columns=[desc.name for desc in cur.description])
            )

            return df
