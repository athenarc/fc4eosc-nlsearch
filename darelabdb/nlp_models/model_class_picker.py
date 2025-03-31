from typing import ClassVar
from darelabdb.nlp_sql_to_text.model import Seq2SeqSql2TextModel
from darelabdb.nlp_text_to_sql.model import Seq2SeqText2SqlModel

SUPPORTED_MODELS = {
    Seq2SeqSql2TextModel.__name__: Seq2SeqSql2TextModel,
    Seq2SeqText2SqlModel.__name__: Seq2SeqText2SqlModel,
}


def model_class_picker(class_name: str):
    """
    Returns the class of dataset on its name.
    Args:
        class_name (str): The name of the requested model class.

    Returns (Dataset): Dataset

    """
    if class_name not in SUPPORTED_MODELS:
        raise NotImplementedError(
            f"The dataset {class_name} is not implemented. The supported datasets "
            f"are: {list(SUPPORTED_MODELS.keys())}"
        )

    return SUPPORTED_MODELS[class_name]
