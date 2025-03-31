import pandas as pd
from loguru import logger

from darelabdb.nlp_metrics.partial_match.components.component_match_ABC import (
    ComponentsMatch,
)
from darelabdb.utils_query_analyzer.query_info.query_info import QueryInfo, CLAUSES


class StructuralMatch(ComponentsMatch):
    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        return [clause.class_name() for clause in CLAUSES] + ["set_operator"]

    @classmethod
    def name(cls):
        return "SM"

    def _get_component_instances(
        self, component: str, elements: list[str]
    ) -> list[str]:
        if component == "set_operator":
            return [
                c
                for c in elements
                if c.startswith("union")
                | c.startswith("intersect")
                | c.startswith("except")
            ]
        else:
            return [c for c in elements if c.startswith(component)]

    def _extract_elements(self, query: QueryInfo) -> list[str]:
        def extract_query_elements(query: QueryInfo) -> list[str]:
            elements = query.structural_components(
                shallow_search=True, with_names=True, with_pos=True
            )

            # Add order in order by
            if "orderby" in elements:
                # Append the orders in the order by in the element
                elements[elements.index("orderby")] = (
                    f"orderby_{'_'.join(query.orderByClause.orders())}"
                )

            return elements

        # If the query is a set operator
        if query.setOperator is not None:
            set_operator_queries = query.setOperator.queries

            # Get the structural components of each query in the set operator
            elements = extract_query_elements(set_operator_queries[0])
            elements2 = extract_query_elements(set_operator_queries[1])

            # Merge the 2 sets of structural components (with an identifier to be distinct)
            elements.extend([f"{element}_1" for element in elements2])

            # Add the set operator
            elements.append(query.setOperator.op)

        else:
            elements = extract_query_elements(query)

        # Differentiate the nestings
        deduplicated_elements = []
        for element in set(elements):
            element_count = elements.count(element)
            if element_count > 1:
                deduplicated_elements.extend(
                    [f"{element}_{num}" for num in range(element_count - 1)]
                )
            deduplicated_elements.append(element)

        return deduplicated_elements

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
