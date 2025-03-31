from typing import List

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts import FewShotPromptFactory
from darelabdb.nlp_text_to_sql_llm.prompts.schemas.fc4eosc import (
    fc4eosc_database_type,
    fc4eosc_schema,
)

# NOTE: We probably do not need them anymore
FEW_SHOT_EXAMPLES = """\
NL question: Find the publications that have more than 100 citations.
SQL query: SELECT * from result as r join result_citations as rc on r.sk_id = rc.sk_result_id_cited WHERE r.type='publication' GROUP BY r.sk_id HAVING COUNT(*) > 100

NL question: Find the publications that mention T5.
SQL query: SELECT * from result as r where r.type = 'publication' and r.description like '%T5%'

NL question: Find the publication with the most authors.
SQL query: SELECT * from result as r join result_author as ra on ra.sk_result_id = r.sk_id where r.type = 'publication' GROUP BY r.sk_id ORDER BY COUNT(*) DESC LIMIT 1
"""


class Fc4eoscPromptFactory(FewShotPromptFactory):
    name: str = "fc4eosc"

    def _get_db_schema_sequence(self, datapoint: SqlQueryDatapoint) -> str:
        return fc4eosc_schema(format=self.schema_type)

    def _get_db_type(self, datapoint: SqlQueryDatapoint) -> str:
        return fc4eosc_database_type()
