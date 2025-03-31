from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


class TextEmbeddingMethod(ABC):
    @abstractmethod
    def get_embeddings(self, sentences: List[str]) -> np.array:
        pass

    @abstractmethod
    def get_items_embedding(self, item_texts: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the embeddings of the items' texts of each sentence. First we unwind the sentences so each row of the
        Dataframe contains only one sentence. Then we create the embeddings of the sentences, and we group them back
        together.

        Example input dataframe:
        ```
        pd.DataFrame([
            [1 ,["Sent 1", "Sent 2"]],
            [2, ["Sent 1", "Sent 2"]]
        ], columns=["id", "sentences"])
        ```

        Args:
            item_texts (pd.DataFrame): The Dataframe with columns "id" and "sentences" (List[str])

        Returns:
            A dataframe with a list of sentence embeddings for each item
        """
        pass

    @abstractmethod
    def get_item_embedding(self, item_text: List[str]) -> List[np.array]:
        """
        Creates the embedding for each sentence of the given item's text.
        """
        pass
