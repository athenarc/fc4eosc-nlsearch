import re
from typing import Dict, List, Literal

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts.PromptFactoryABC import PromptFactory
from darelabdb.nlp_text_to_sql_llm.prompts.templates import (
    INSTRUCTIONS_TEMPLATE,
    QUESTION_TEMPLATE,
)

PROMPT_TEMPLATE = (
    INSTRUCTIONS_TEMPLATE + "\n\n" + QUESTION_TEMPLATE + "\n\n"
)


class ZeroShotPromptFactory(PromptFactory):
    name: str = "zero_shot_prompt"

    def __init__(
        self,
        schema_type: Literal["ddl", "compact"] = "ddl",
        include_pk: bool = True,
        include_fk: bool = True,
        include_notes: bool = True,
        values_num: int = 0,
        categorical_threshold: int = None,
    ) -> None:
        super().__init__(
            schema_type,
            include_pk,
            include_fk,
            include_notes,
            values_num,
            categorical_threshold,
        )

    def prompt_template(self):
        return PROMPT_TEMPLATE

    def fill_prompt(self, datapoint: SqlQueryDatapoint) -> str:
        """Fills the prompt string template for the given datapoint.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used as a
                question for the LLM.

        Returns:
            str: The prompt for the LLM as a string.
        """
        schema_sequence = self._get_db_schema_sequence(datapoint)
        database_type = self._get_db_type(datapoint)

        return self.prompt_template().format(
            nl_query=datapoint.nl_query,
            database_type=database_type,
            database_schema=schema_sequence,
        )

    def fill_chat(self, datapoint: SqlQueryDatapoint) -> List[Dict[str, str]]:
        """Fills the chat template for the given datapoint.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used as a
                question for the LLM.

        Returns:
            List[Dict[str, str]]: The chat for the LLM as a list of dictionaries.
                Each dictionary has two keys `role` and `content`.
        """
        schema_sequence = self._get_db_schema_sequence(datapoint)
        database_type = self._get_db_type(datapoint)
        chat = [
            {
                "role": "system",
                "content": INSTRUCTIONS_TEMPLATE.format(
                    database_type=database_type, database_schema=schema_sequence
                ),
            },
            {
                "role": "user",
                "content": QUESTION_TEMPLATE.format(nl_query=datapoint.nl_query),
            },
        ]
        return chat

    def extract_sql(self, response: str) -> str:
        """Extract sql from a markdown fenced code block.

        The function will return everything inside a fenced code block, taking
        into account that there might be a "sql" language specification next to
        the opening backticks.
        If the extraction fails, the function returns the given response.
        """
        # NOTE: The *? symbol matches as little text as possible
        pattern_recipe = r"```(sql)?([\S\s]*?)```"
        match = re.search(pattern_recipe, response, flags=re.IGNORECASE)

        if match is None:
            return response
        match_text = match.group(2)

        return match_text
