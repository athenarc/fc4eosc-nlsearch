from loguru import logger

from darelabdb.nlp_embeddings.embedding_methods.SBERTEmbedding import SBERTEmbedding
from darelabdb.nlp_embeddings.embedding_methods.TextEmbeddingMethodABC import (
    TextEmbeddingMethod,
)


def pick_embedding_method(embedding_method_name: str, **kwargs) -> TextEmbeddingMethod:
    match embedding_method_name:
        case SBERTEmbedding.name:
            return SBERTEmbedding(**kwargs)
        case _:
            logger.error(
                f"The embedding method '{embedding_method_name}' was not found."
            )
            raise AttributeError(
                f"The embedding method '{embedding_method_name}' was not found."
            )
