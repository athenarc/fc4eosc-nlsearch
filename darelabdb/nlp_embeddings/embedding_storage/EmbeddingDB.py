from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import pandas as pd


class EmbeddingDB(ABC):

    def __init__(
        self, db_name: str, embedding_col_name: str, primary_key_col_name: str = None
    ):
        """
        Initialize the EmbeddingDB object.

        Args:
            db_name: The name of the database.
            embedding_col_name: The name of the column that will be used to store the embeddings.
                When inserting the rows, this name should appear in the rows dictionary.
            primary_key_col_name: (Optional) The name of the column that will be used as the primary key.
                When inserting the rows, this name should appear in the rows dictionary.
        """
        self.db_name = db_name
        self.primary_key_col_name = primary_key_col_name
        self.embedding_col_name = embedding_col_name

        if self.embedding_col_name is None:
            raise ValueError("The name of the embedding column is required.")

        if self.primary_key_col_name == self.embedding_col_name:
            raise ValueError(
                f"The primary key column name and the embedding column name must be different."
            )

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def populate(self, rows: Dict, **kwargs) -> None:
        """
        Insert multiple item embeddings into the database.

        Args:
            rows: A dictionary where the keys are the column names and the values are the values of the
                columns.
        """
        pass

    @abstractmethod
    def get_embedding(self, row_id: str) -> Optional[List[float]]:
        """
        Get the embedding of the given item.

        Args:
            row_id: The id (primary key) of the row that we want to get the embedding of.

        Returns:
            The embedding of the item or None if a primary key column was not specified upon initialization.
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        embedding: List[float],
        num: int,
        eq_filters: dict = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the nearest neighbors of the given embedding.

        Args:
            embedding: An embedding of an item that we want to find the neighbors of.
            num: The number of neighbors that we want to get.
            eq_filters: A dictionary with {`column_name`: `value`} to be applied, to exclude rows.
            columns: The columns that will be returned in the result.

        Returns:
            A pandas DataFrame with the columns specified in the columns parameter and the similarity score.
        """
        pass
