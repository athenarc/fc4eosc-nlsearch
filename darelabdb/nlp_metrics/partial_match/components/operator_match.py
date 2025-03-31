from loguru import logger
import pandas as pd

from darelabdb.nlp_metrics.partial_match.components.component_match_ABC import (
    ComponentsMatch,
)
from darelabdb.utils_query_analyzer.query_info.query_info import (
    QueryInfo,
    OPERATOR_TYPES,
)


class OperatorsMatch(ComponentsMatch):
    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        comps = []
        for operator_type in OPERATOR_TYPES:
            comps.extend(operator_type.members())
        return comps

    @classmethod
    def name(cls):
        return "OM"

    def _get_component_instances(
        self, component: str, elements: list[str]
    ) -> list[str]:
        return [e for e in elements if e.startswith(component)]

    def _extract_elements(self, query: QueryInfo) -> list[str]:
        operators_per_clause = query.operators(shallow_search=True, per_clause=True)

        elements = []
        for clause, operators in operators_per_clause.items():
            for operator in operators:
                num = sum(f"{operator}_{clause}" in e for e in elements)
                elements.append(f"{operator}_{clause}_{num}")

        return elements

    def _calculate_errors(
        self, ref_elements: list[str], pred_elements: list[str], depth: int = None
    ) -> None:
        """The calculation of errors is not implemented."""
        pass

    def error_types(self) -> pd.DataFrame:
        logger.warning(
            "The operator match does not have information about the error types."
        )
        return pd.DataFrame()
