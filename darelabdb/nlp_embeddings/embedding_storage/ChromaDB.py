import json
from typing import List, Optional, Dict

import pandas as pd
from loguru import logger
import chromadb
from chromadb.errors import InvalidCollectionException

from darelabdb.nlp_embeddings.embedding_storage.EmbeddingDB import EmbeddingDB


# TODO add a parameter for 'remove if exist' in the initialization of a collection
class ChromaDB(EmbeddingDB):

    def __init__(
        self, db_name: str, embedding_col_name: str, primary_key_col_name: str
    ):
        """
        Initialize the EmbeddingDB object.

        Args:
            db_name: The name of the database.
            embedding_col_name: The name of the column that will be used to store the embeddings.
                When inserting the rows, this name should appear in the rows dictionary.
            primary_key_col_name: The name of the column that will be used as the primary key.
                When inserting the rows, this name should appear in the rows dictionary.
        """
        super().__init__(db_name, embedding_col_name, primary_key_col_name)

        self.client = chromadb.Client()
        self.collection = None
        self.initialize()
        self._serialized_columns = []

    def initialize(self) -> None:
        """
        Initialize the collection that will be used to store the embeddings.
        """
        if self.collection is None:
            try:
                self.collection = self.client.get_collection(self.db_name)
            except InvalidCollectionException:
                self.collection = self.client.create_collection(
                    name=self.db_name, metadata={"hnsw:space": "cosine"}
                )

    def is_populated(self) -> bool:
        return self.collection.count() != 0

    def populate(self, rows: Dict) -> None:
        """
        Insert multiple item embeddings into the database.

        Args:
            rows: A dictionary where the keys are the column names and the values are the values of the
                columns. e.g.,
                {
                    "value_col": ["a", "b", "c"],
                    "embedding_col": [[0.1, 0.2], [0.3, 0.1], [0.6, 0.4]]
                }
        """
        # Get the embedding column and the primary key column
        embedding_col_values = rows.pop(self.embedding_col_name)
        primary_key_col_values = rows.pop(self.primary_key_col_name)

        # Convert to json-serialized string the values of any columns that contain dictionaries
        for column, values in rows.items():
            if type(values[0]) not in [str, int, float, bool]:
                self._serialized_columns.append(column)
                rows[column] = [json.dumps(value) for value in values]

        # Transform the dictionary into a list of dicts
        records_metadata = [dict(zip(rows, values)) for values in zip(*rows.values())]

        self.initialize()

        self.collection.add(
            embeddings=embedding_col_values,
            metadatas=records_metadata,
            ids=primary_key_col_values,
        )

    def get_embedding(self, row_id: str) -> Optional[List[float]]:
        """
        Get the embedding of the given item.

        Args:
            row_id: The id (primary key) of the row that we want to get the embedding of.

        Returns:
            The embedding of the item or None of the id was not found.
        """
        if self.collection is None:
            logger.warning("There are no entries in the vector database.")
            return None

        results = self.collection.get(ids=[row_id], include=["embeddings"])

        if len(results["embeddings"]) == 0:
            logger.info(f"There are no entries in the vector database for id {row_id}.")
            return None

        return list(results["embeddings"][0])

    def get_neighbors(
        self,
        embedding: List[float],
        num: int,
        eq_filters: dict = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the neighbors of the given embedding using the cosine similarity operator.
        Args:
            embedding: An embedding of an item that we want to find the neighbors of.
            num: The number of neighbors that we want to get.
            eq_filters: A dictionary with {<column_name>: <value>} to be applied, to exclude rows.
            columns: The columns that will be returned in the result.

        Returns:
            A pandas DataFrame with the columns specified in the columns parameter and the similarity score.
        """
        if self.collection is None:
            logger.warning("There are no entries in the vector database.")
            return pd.DataFrame({})

        results = self.collection.query(
            query_embeddings=[embedding], n_results=num, where=eq_filters
        )

        result_rows = []
        for row_id, metadata, distance in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            result_rows.append(
                {
                    self.primary_key_col_name: row_id,
                    "similarity": 1.0 - distance,
                    **metadata,
                }
            )

        result_rows_df = pd.DataFrame(result_rows)

        # Convert to dict any json-serializable columns
        for column in self._serialized_columns:
            if column in result_rows_df.columns:
                result_rows_df[column] = result_rows_df[column].apply(
                    lambda x: json.loads(x)
                )

        if columns is not None:
            result_rows_df = result_rows_df[columns + ["similarity"]]

        if result_rows_df.shape[0] < num:
            logger.warning(
                "There are less entries in the vector database than the requested number of neighbors. All"
                "the records have been returned"
            )

        return result_rows_df
