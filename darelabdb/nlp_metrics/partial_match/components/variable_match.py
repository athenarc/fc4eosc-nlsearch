from abc import ABC
from loguru import logger
import pandas as pd

from darelabdb.nlp_metrics.partial_match.components.component_match_ABC import (
    ComponentsMatch,
)
from darelabdb.utils_query_analyzer.query_info.query_info import (
    QueryInfo,
    VARIABLE_TYPES,
)


# TODO normalize the columns and tables in the query


class VariablesMatch(ComponentsMatch):
    def __init__(
        self, reference: QueryInfo, prediction: QueryInfo, with_values: bool = True
    ):
        self.with_values = with_values
        self.error_types_per_schema_element = {
            "error_type": [],
            "variable_name": [],
            "variable_type": [],
            "depth": [],
        }
        super().__init__(reference, prediction)

    @classmethod
    def components(cls) -> list[str]:
        return VARIABLE_TYPES

    @classmethod
    def name(cls):
        return "VM"

    def _get_component_instances(
        self, component: str, elements: list[str]
    ) -> list[str]:
        return [e for e in elements if e.startswith(component + "/")]

    def _extract_elements(self, query: QueryInfo) -> list[str]:
        variable_types = (
            self.components()
            if self.with_values
            else self.components().remove("values")
        )

        self._columns_equivalences = query.get_columns_equivalences()

        elements = []
        for variable_type in variable_types:
            query_variables = (
                getattr(query, variable_type)(shallow_search=True)
                if variable_type != "columns"
                else getattr(query, "columns_without_table_aliases")(
                    shallow_search=True
                )
            )
            for variable in query_variables:
                # append element <variable_type>/<variable>/<i>
                new_element = f"{variable_type}/{str(variable)}"
                elements.append(
                    new_element
                    + "/"
                    + str(sum(new_element + "/" in e for e in elements))
                )
        return elements

    def _replace_equivalence(
        self, set1: set[str], set2: set[str]
    ) -> (set[str], set[str], bool):
        if len(self._columns_equivalences) == 0:
            return set1, set2, False

        set1_column_difference = {
            var for var in set1.difference(set2) if "columns/" in var
        }
        set2_column_difference = {
            var for var in set2.difference(set1) if "columns/" in var
        }

        found_equivalence = False
        for column1 in sorted(set1_column_difference, reverse=True):
            _, column_name1, number1 = column1.split("/")
            # If this is not the first appearance of the column (must appear at least one for the equivalence to exist)
            # and there exist an equivalence column
            if int(number1) > 0 and column_name1 in self._columns_equivalences:
                # search for the equivalent
                for column2 in sorted(set2_column_difference):
                    _, column_name2, number2 = column2.split("/")
                    if (
                        int(number2) > 0
                        and column_name2 in self._columns_equivalences[column_name1]
                    ):
                        found_equivalence = True
                        set1.remove(column1)
                        # append element <variable_type>/<variable>/<i>
                        new_element = "columns/" + str(column_name2).lower()
                        set1.add(
                            new_element
                            + "/"
                            + str(sum(new_element + "/" in e for e in set1))
                        )
                        break
        return set1, set2, found_equivalence

    def _similarity(self, ref_elements: list[str], pred_elements: list[str]) -> float:
        ref_elements = set(ref_elements)
        pred_elements = set(pred_elements)

        equivalence_detected = True
        while equivalence_detected:
            (
                ref_elements,
                pred_elements,
                equivalence_detected,
            ) = self._replace_equivalence(ref_elements, pred_elements)

        return super()._similarity(list(ref_elements), list(pred_elements))

    def _calculate_errors(
        self, ref_elements: list[str], pred_elements: list[str], depth: int = None
    ) -> None:
        """
        Calculates the false positive and false negative predictions of the schema elements and updates the
            `error_types_per_schema_element`.

        Args:
            ref_elements (list[str]): The list of the elements existing in the referenced subquery
            pred_elements (list[str]): The list of the elements existing in the predicted subquery
            depth (int): The depth of the subquery. If None, the errors will not store information about the depth.

        """

        pred_elements_copy = pred_elements.copy()

        # For every referenced element
        for element in ref_elements:
            # If the element is not in the predicted elements, it is a false negative
            if element not in pred_elements_copy:
                self.error_types_per_schema_element["error_type"].append("FN")
                self.error_types_per_schema_element["variable_name"].append(
                    element.split("/")[1]
                )
                self.error_types_per_schema_element["variable_type"].append(
                    element.split("/")[0]
                )
            else:
                # Remove the element from the predicted elements
                pred_elements_copy.remove(element)

        # Every element left in the predicted elements is a false positive
        for element in pred_elements_copy:
            self.error_types_per_schema_element["error_type"].append("FP")
            self.error_types_per_schema_element["variable_name"].append(
                element.split("/")[1]
            )
            self.error_types_per_schema_element["variable_type"].append(
                element.split("/")[0]
            )

        # Store the information about depth
        self.error_types_per_schema_element["depth"].extend(
            [depth] * len(self.error_types_per_schema_element["error_type"])
        )

    def error_types(self, depth: int = None) -> pd.DataFrame:
        """
        Returns the error types of the schema elements.
        Args:
            depth (int): If None, all the errors will be returned. Otherwise, the errors of the specific depth will be
                returned.

        Returns (pd.DataFrame): A dataframe with columns = ["error_type", "variable_name", "variable_type"]
        """

        errors_df = pd.DataFrame(self.error_types_per_schema_element)

        if depth is not None:
            # If the requested depth is not in the errors or all depths are None
            if depth not in errors_df["depth"] or errors_df["depth"].isnull().all():
                logger.warning(
                    f"There are no information about errors in depth {depth}. "
                    f"All the errors will be returned instead."
                )
            else:
                # Return the errors of the specific depth
                return errors_df[errors_df["depth"] == depth].drop(columns="depth")

        # Return the errors without the depth column
        return errors_df.drop(columns="depth")
