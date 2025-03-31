from abc import ABC, abstractmethod
import pandas as pd

from darelabdb.nlp_embeddings.embedding_methods.TextEmbeddingMethodABC import (
    TextEmbeddingMethod,
)
from darelabdb.nlp_embeddings.embedding_storage.EmbeddingDB import EmbeddingDB


class SimilarExamplesSelector(ABC):
    """The class responsible for the selection of similar nl-sql examples based on the given nl question."""

    @abstractmethod
    def __init__(
        self,
        embedding_db: EmbeddingDB,
        text_embedding_method: TextEmbeddingMethod = None,
        skip_n: int = 0,
    ):
        """
        Args:
            embedding_db: The database that stores the embeddings of the examples.
            text_embedding_method: The method to embed the text of the examples.
            skip_n: The number of similar examples to skip. Usually set to 1 for evaluation to avoid include in the
            similar example the query itself.
        """
        pass

    @abstractmethod
    def get_similar_examples(self, nl_question: str, num: int) -> pd.DataFrame:
        """
        Retrieves similar examples to the one provided.

        Args:
            nl_question: The natural language question of the example for which we want to retrieve similar examples.
            num: The number of similar examples to return.

        Returns:
            A dataframe containing the similar examples. columns=['nl_question', 'sql_query']
        """
        pass
