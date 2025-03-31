from typing import List, Optional

import numpy as np
import pandas as pd
from darelabdb.nlp_embeddings.embedding_methods.TextEmbeddingMethodABC import (
    TextEmbeddingMethod,
)
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class SBERTEmbedding(TextEmbeddingMethod):
    name = "sbert_embedding"
    model: Optional[SentenceTransformer]

    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 1,
        model_persistence: bool = False,
        dimension: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.dimension = dimension

        self.model = (
            None
            if not model_persistence
            else SentenceTransformer(self.model_name, device=self.device)
        )

    def get_embeddings(
        self, sentences: List[str], show_progress_bar: bool = True
    ) -> np.array:
        model = (
            self.model
            if self.model is not None
            else SentenceTransformer(self.model_name, device=self.device)
        )
        return model.encode(
            sentences, show_progress_bar=show_progress_bar, batch_size=self.batch_size
        )

    def _dimensionality_reduction(self, embeddings: np.array) -> np.array:
        if self.dimension >= min(embeddings.shape[1], embeddings.shape[0]):
            logger.warning(
                f"Dimension {self.dimension} is greater or equal than the original dimension {embeddings.shape[1]} "
                f"or the number of items {embeddings.shape[0]}. Returning the original embeddings."
            )
            return embeddings

        pca = PCA(n_components=self.dimension)
        embeddings = pca.fit_transform(embeddings)

        return embeddings

    def get_items_embedding(self, item_texts: pd.DataFrame) -> pd.DataFrame:
        item_texts = item_texts.explode("sentences")
        item_texts["sentences"] = item_texts["sentences"].fillna("")
        sentence_embeddings = self.get_embeddings(item_texts["sentences"].tolist())
        if self.dimension is not None:
            sentence_embeddings = self._dimensionality_reduction(sentence_embeddings)

        item_texts["sentence_embeddings"] = sentence_embeddings.tolist()
        item_texts = item_texts.groupby("id").agg(list).reset_index()

        return item_texts

    def get_item_embedding(self, item_text: List[str]) -> List[np.array]:
        # NOTE: For a single item we don't display the progress bar to avoid too many messages
        return self.get_embeddings(item_text, show_progress_bar=False)
