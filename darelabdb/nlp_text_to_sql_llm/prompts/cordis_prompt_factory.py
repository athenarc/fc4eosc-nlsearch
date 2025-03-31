from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts import ZeroShotPromptFactory
from darelabdb.nlp_text_to_sql_llm.prompts.schemas.cordis import (
    cordis_database_type,
    cordis_schema,
)


class CordisPromptFactory(ZeroShotPromptFactory):
    name: str = "cordis"

    def _get_db_schema_sequence(self, datapoint: SqlQueryDatapoint) -> str:
        return cordis_schema(format=self.schema_type)

    def _get_db_type(self, datapoint: SqlQueryDatapoint) -> str:
        return cordis_database_type()
