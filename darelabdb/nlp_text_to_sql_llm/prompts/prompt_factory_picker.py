from typing import Dict

from darelabdb.nlp_text_to_sql_llm.prompts import (
    CordisPromptFactory,
    Fc4eoscPromptFactory,
    FewShotPromptFactory,
    PromptFactory,
    ZeroShotPromptFactory,
)
from loguru import logger


def pick_prompt_factory(prompt_version: str, **prompt_arguments) -> PromptFactory:

    match prompt_version:
        case ZeroShotPromptFactory.name:
            return ZeroShotPromptFactory(**prompt_arguments)
        case FewShotPromptFactory.name:
            return FewShotPromptFactory(**prompt_arguments)
        case Fc4eoscPromptFactory.name:
            return Fc4eoscPromptFactory(**prompt_arguments)
        case CordisPromptFactory.name:
            return CordisPromptFactory(**prompt_arguments)
        case _:
            logger.error(f"Prompt version {prompt_version} not found.")
            raise AttributeError(f"Prompt version {prompt_version} not found.")
