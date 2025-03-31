from abc import ABC, abstractmethod
from typing import Dict, List, Literal

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts.db_schema_sequence import (
    get_db_schema_sequence,
)


class PromptFactory(ABC):
    @abstractmethod
    def __init__(
        self,
        schema_type: Literal["ddl", "compact"],
        include_pk: bool,
        include_fk: bool,
        include_notes: bool,
        values_num: int = 0,
        categorical_threshold: int = None,
    ) -> None:
        # DB schema sequence parameters
        self.schema_type = schema_type
        self.include_pk = include_pk
        self.include_fk = include_fk
        self.include_notes = include_notes
        self.values_num = values_num
        self.categorical_threshold = categorical_threshold

    def _get_db_schema_sequence(self, datapoint: SqlQueryDatapoint) -> str:
        """Creates the DB schema sequence for the given datapoint."""
        return get_db_schema_sequence(
            datapoint.db_schema,
            type=self.schema_type,
            include_pk=self.include_pk,
            include_fk=self.include_fk,
            include_notes=self.include_notes,
            values_num=self.values_num,
            categorical_threshold=self.categorical_threshold,
            db_id=datapoint.db_id
        )

    def _get_db_type(self, datapoint: SqlQueryDatapoint) -> str:
        # TODO: Is this the best way to get the DB type?
        return "SQLite" if datapoint.db_path.endswith(".sqlite") else "Postgres"

    @abstractmethod
    def prompt_template(self) -> str:
        """The template of the prompt, that contains the description."""
        pass

    @abstractmethod
    def fill_prompt(self, datapoint: SqlQueryDatapoint, *args, **kwargs) -> str:
        """Fills the prompt template with the given arguments and returns a prompt.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used to
                fill the template.
            args: Additional arguments passed to the template (e.g., the database schema).
            **kwargs: Additional keyword arguments passed to the template
                (e.g., the database schema).

        Returns:
            str: The prompt template filled with the given arguments.
        """
        pass

    @abstractmethod
    def fill_chat(
        self, datapoint: SqlQueryDatapoint, *args, **kwargs
    ) -> List[Dict[str, str]]:
        """Fills the prompt template with the given arguments and returns a chat.

        Args:
            datapoint (SqlQueryDatapoint): The datapoint that will be used to
                fill the template.
            args: Additional arguments passed to the template (e.g., the database schema).
            **kwargs: Additional keyword arguments passed to the template
                (e.g., the database schema).

        Returns:
            List[Dict[str, str]]: The chat version of the prompt template filled
                with the given arguments.
        """
        pass

    @abstractmethod
    def extract_sql(self, response: str) -> str:
        """Extracts the SQL query from the given response.

        Args:
            response: The response of the model.

        Returns:
            The SQL query in the response.
        """
        pass
