import pandas as pd
from darelabdb.nlp_embeddings.embedding_methods.SBERTEmbedding import SBERTEmbedding
from darelabdb.nlp_embeddings.embedding_methods.TextEmbeddingMethodABC import (
    TextEmbeddingMethod,
)
from darelabdb.nlp_embeddings.embedding_storage.EmbeddingDB import EmbeddingDB
from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_examples_selector_ABC import (
    SimilarExamplesSelector,
)


class SimilarNLSelector(SimilarExamplesSelector):
    """
    The class responsible for the selection of similar natural language questions, given a corpus of nl questions and
    sql queries.
    """

    name = "similar_nl_selector"

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
            skip_n: The number of similar examples to skip. Usually set to 1 for evaluation
        """
        self.text_embedding_method = (
            text_embedding_method
            if text_embedding_method is not None
            else SBERTEmbedding()
        )
        self.embedding_db = embedding_db
        self.skip_n = skip_n

    def update(self, example_pool: pd.DataFrame) -> None:
        """
        Creates and stores the embeddings of the provided examples.

        Args:
            example_pool: The available corpus from which the similar examples will be selected. The dataframe must
                            contain the columns 'nl_question' and 'sql_query'.
        """
        # Create the embeddings of the given corpus
        embeddings = [
            self.text_embedding_method.get_item_embedding([item_text])[0].tolist()
            for item_text in example_pool["nl_question"].values.tolist()
        ]

        # Store the embeddings
        self.embedding_db.insert_many(
            column_types={
                "nl_question": "TEXT",
                "sql_query": "TEXT",
                "embedding": "vector",
            },
            rows={
                "nl_question": example_pool["nl_question"].values.tolist(),
                "sql_query": example_pool["sql_query"].values.tolist(),
                "embedding": embeddings,
            },
        )

    def get_similar_examples(
        self, nl_question: str, num: int, eq_filters: dict = None
    ) -> pd.DataFrame:
        """
        Returns the most similar example pairs of nl questions and sql queries based on the similarity between the
            given nl_question and the ones existing in the corpus.

        Args:
            nl_question: The question for which similar examples will be selected.
            num: The number of similar examples to return.
            eq_filters: A dictionary with the names of the columns and the values that we want to apply as a filter in the
            search. (e.g., eq_filter = {db_id: 1})

        Returns:
            A dataframe with the most similar examples. The dataframe contains the columns=['nl_question', 'sql_query']
        """

        # Try to find the embedding in the database
        embedding = self.embedding_db.get_embedding(row_id=nl_question)
        if embedding is None:
            # Calculate the embedding of the given nl_question
            embedding = self.text_embedding_method.get_item_embedding([nl_question])[
                0
            ].tolist()

        # Find the most similar embeddings in the corpus
        return self.embedding_db.get_neighbors(
            embedding, num=num + self.skip_n, eq_filters=eq_filters
        )[self.skip_n :]
