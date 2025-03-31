from loguru import logger

from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_examples_selector_ABC import (
    SimilarExamplesSelector,
)
from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_nl_selector import (
    SimilarNLSelector,
)


def pick_similar_examples_selector(
    selector_name: str, **example_selector_args
) -> SimilarExamplesSelector:
    match selector_name:
        case SimilarNLSelector.name:
            return SimilarNLSelector(**example_selector_args)
        case _:
            logger.error(
                f"The selector for the similar examples '{selector_name}' was not found."
            )
            raise AttributeError(
                f"The selector for the similar examples '{selector_name}' was not found."
            )
