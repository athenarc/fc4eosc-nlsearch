import re
from typing import Dict, List, Literal

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts.PromptFactoryABC import PromptFactory
from darelabdb.nlp_text_to_sql_llm.prompts.templates import (
    CHAT_SQL_TEMPLATE,
    INSTRUCTIONS_TEMPLATE,
    PROMPT_SIMILAR_EXAMPLES_TEMPLATE,
    QUESTION_TEMPLATE,
)

PROMPT_TEMPLATE = (
    INSTRUCTIONS_TEMPLATE
    + "\n\n"
    + PROMPT_SIMILAR_EXAMPLES_TEMPLATE
    + "\n\n"
    + QUESTION_TEMPLATE
)

class FewShotPromptFactory(PromptFactory):
    name: str = "few_shot_prompt"

    def __init__(
        self,
        schema_type: Literal["ddl", "compact"] = "ddl",
        include_pk: bool = True,
        include_fk: bool = True,
        include_notes: bool = True,
        values_num: int = 0,
        categorical_threshold: int = None,
        similar_examples_representation: Literal[
            "with_schema", "without_schema"
        ] = "without_schema",
    ) -> None:
        super().__init__(
            schema_type,
            include_pk,
            include_fk,
            include_notes,
            values_num,
            categorical_threshold,
        )

        self.similar_examples_representation = similar_examples_representation

    def prompt_template(self):
        return PROMPT_TEMPLATE

    def fill_prompt(
        self, datapoint: SqlQueryDatapoint, examples: List[SqlQueryDatapoint]
    ) -> str:
        """Fills the prompt string template for the given datapoint and examples.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used as a
                question for the LLM.
            examples (List[SqlQueryDatapoint]): The datapoints that will be used
                as few-shot examples for the LLM.

        Returns:
            str: The prompt for the LLM as a string.
        """
        schema_sequence = self._get_db_schema_sequence(datapoint)
        database_type = self._get_db_type(datapoint)

        examples_sequence = ""
        for example_index in range(len(examples)):
            example = examples[example_index]

            examples_sequence += f"#{example_index}.\n\n"
            if self.similar_examples_representation == "with_schema":
                examples_sequence += (
                    f"Database Schema: {self._get_db_schema_sequence(example)}\n"
                )
            examples_sequence += f"NL Question:\n{example.nl_query}\n\n"
            examples_sequence += f"SQL Query:\n{example.sql_query}\n\n"

        return self.prompt_template().format(
            nl_query=datapoint.nl_query,
            database_type=database_type,
            database_schema=schema_sequence,
            examples=examples_sequence,
        )

    def fill_chat(
        self, datapoint: SqlQueryDatapoint, examples: List[SqlQueryDatapoint]
    ) -> List[Dict[str, str]]:
        """Fills the chat template for the given datapoint and examples.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used as a
                question for the LLM.
            examples (List[SqlQueryDatapoint]): The datapoints that will be used
                as few-shot examples for the LLM.

        Returns:
            List[Dict[str, str]]: The chat for the LLM as a list of dictionaries.
                Each dictionary has two keys `role` and `content`.
        """
        chat = []

        schema_sequence = self._get_db_schema_sequence(datapoint)
        database_type = self._get_db_type(datapoint)

        chat.append(
            {
                "role": "system",
                "content": INSTRUCTIONS_TEMPLATE.format(
                    database_type=database_type, database_schema=schema_sequence
                ),
            }
        )

        # Add few-shot examples
        for example in examples:
            user_content = ""
            if self.similar_examples_representation == "with_schema":
                user_content += (
                    f"Database schema: {self._get_db_schema_sequence(example)}\n"
                )
            user_content += QUESTION_TEMPLATE.format(nl_query=example.nl_query)
            chat.append(
                {
                    "role": "user",
                    "content": user_content,
                }
            )

            chat.append(
                {
                    "role": "assistant",
                    "content": CHAT_SQL_TEMPLATE.format(sql_query=example.sql_query),
                }
            )

        # Add the test question
        chat.append(
            {
                "role": "user",
                "content": QUESTION_TEMPLATE.format(nl_query=datapoint.nl_query),
            }
        )

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
