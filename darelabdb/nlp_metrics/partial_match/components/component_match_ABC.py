from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from darelabdb.utils_query_analyzer.query_info.query_info import (
    QueryInfo,
    get_subqueries_per_depth,
)


class ComponentsMatch(ABC):
    """Calculates the match for a set of components."""

    def __init__(self, reference: QueryInfo, prediction: QueryInfo):
        """
        Initializes the component match.
        """
        self.matches = self._calc(reference, prediction)

    @classmethod
    @abstractmethod
    def components(cls) -> list[str]:
        pass

    @abstractmethod
    def _extract_elements(self, query: QueryInfo) -> list[str]:
        """Returns a list with the elements existing in the query regarding the components."""
        pass

    @abstractmethod
    def _get_component_instances(
        self, component: str, elements: list[str]
    ) -> list[str]:
        """Returns a sublist of the elements that contains only the component instances."""
        pass

    @staticmethod
    def _get_max_subqueries_per_depth(
        ref_subqueries: dict[int, list[QueryInfo]],
        pred_subqueries: dict[int, list[QueryInfo]],
    ) -> list[int]:
        # Calculate max depth
        max_depth = max(
            max(list(ref_subqueries.keys())), max(list(pred_subqueries.keys()))
        )

        max_subqueries_per_depth = []
        for depth in range(max_depth + 1):
            if len(ref_subqueries) > depth and len(pred_subqueries) > depth:
                max_subqueries_per_depth.append(
                    max(len(ref_subqueries[depth]), len(pred_subqueries[depth]))
                )
            elif len(ref_subqueries) <= depth:
                max_subqueries_per_depth.append(len(pred_subqueries[depth]))
            else:  # len(pred_subqueries) <= depth:
                max_subqueries_per_depth.append(len(ref_subqueries[depth]))
        return max_subqueries_per_depth

    @staticmethod
    def _jaccard_similarity(a: list[str], b: list[str]) -> float:
        a = set(a)
        b = set(b)
        intersect_len = len(a.intersection(b))
        union_len = len(a.union(b))
        if union_len > 0:
            return float(intersect_len) / union_len
        else:
            return 1

    @staticmethod
    def _index_exist(data: dict[int, list[Any]], index1: int, index2: int) -> bool:
        """Returns True if in the given list there is the dimension data[index1][index2], False otherwise."""
        if index1 in data and len(data[index1]) > index2:
            return True
        else:
            return False

    def _similarity(self, ref_elements: list[str], pred_elements: list[str]) -> float:
        """Returns the similarity score of 2 lists."""
        return self._jaccard_similarity(ref_elements, pred_elements)

    @abstractmethod
    def _calculate_errors(
        self, ref_elements: list[str], pred_elements: list[str], depth: int = None
    ) -> None:
        """
        Calculates the errors that appear in the predicted elements. For example the false positive predictions of a
            specific clause.

        Args:
            ref_elements (list[str]): The list of the elements existing in the referenced subquery
            pred_elements (list[str]): The list of the elements existing in the predicted subquery
            depth (int): The depth of the subquery. If None, the errors will not store information about the depth.
        """
        pass

    def _get_components_similarities(
        self, ref_elements: list[str], pred_elements: list[str]
    ) -> dict[str, float]:
        """
        Returns the similarity of every component
        Args:
            ref_elements (list[str]): The list of the elements existing in the referenced subquery
            pred_elements (list[str]): The list of the elements existing in the predicted subquery

        Returns (dict[str, float]): A dictionary with the match value of every component
        """

        matches = {}

        # For every component
        for component in self.components():
            # Compute the component's similarity
            similarity = self._similarity(
                self._get_component_instances(component, ref_elements),
                self._get_component_instances(component, pred_elements),
            )
            matches[component] = similarity

        return matches

    def _calc(
        self, reference: QueryInfo, prediction: QueryInfo
    ) -> dict[int, dict[str, float]]:
        """
        Calculates the component match of 2 queries and the errors occured (e.g., the false positive and false negative
            errors in each component).

        Args:
            reference (QueryInfo): The reference query.
            prediction (QueryInfo): The predicted query.

        Returns (dict[int, dict], dict): A tuple with
            - a dictionary with the total and each component match value in every depth and
            - a dictionary with the errors that occurred in the prediction.
        """

        # Get the subqueries of each depth
        ref_subqueries = get_subqueries_per_depth(reference)
        pred_subqueries = get_subqueries_per_depth(prediction)

        max_subqueries_per_depth = self._get_max_subqueries_per_depth(
            ref_subqueries, pred_subqueries
        )

        # Initialize the matches dictionary
        matches = {d: {} for d in range(len(max_subqueries_per_depth))}

        # For every depth
        for depth, max_subqueries_num in enumerate(max_subqueries_per_depth):
            # Initialize the depth similarities
            depth_similarities = {k: [] for k in ["total"] + self.components()}

            # Compare the subqueries of the current depth
            for i in range(max_subqueries_num):
                # Check that the i-th subquery in the current depth exist both in the reference and in prediction
                ref_exist = self._index_exist(ref_subqueries, depth, i)
                pred_exist = self._index_exist(pred_subqueries, depth, i)

                # Extract the elements of each subqueries
                ref_elements = (
                    self._extract_elements(ref_subqueries[depth][i])
                    if ref_exist
                    else []
                )
                pred_elements = (
                    self._extract_elements(pred_subqueries[depth][i])
                    if pred_exist
                    else []
                )

                # Calculate the errors in the components
                self._calculate_errors(ref_elements, pred_elements, depth)

                # Calculate the total components similarity of the 2 subqueries
                depth_similarities["total"].append(
                    self._similarity(ref_elements, pred_elements)
                )

                # Calculate the similarity of each component in the subquery
                for (
                    component,
                    component_similarity,
                ) in self._get_components_similarities(
                    ref_elements, pred_elements
                ).items():
                    depth_similarities[component].append(component_similarity)

                # Calculate the total and components match for the current depth
                matches[depth]["total"] = sum(depth_similarities["total"]) / len(
                    depth_similarities["total"]
                )
                for component in self.components():
                    matches[depth][component] = sum(
                        depth_similarities[component]
                    ) / len(depth_similarities[component])

        return matches

    def value(self, depth: int = None, component: str = None) -> float:
        """
        Returns the requested match value.
        Args:
            depth (int): The depth that will be considered in the match value returned. If None, the average of all
                depths will be returned.
            component (str): The name of the component that will be considered in the match value returned. If None,
                the total match will be returned.
        """

        if depth is None and component is None:
            # Return the total match for all depths
            return sum(
                [depth_matches["total"] for _, depth_matches in self.matches.items()]
            ) / len(self.matches)
        elif component is not None and depth is not None:
            return self.matches[depth][component]
        elif depth is not None:
            return self.matches[depth]["total"]
        else:  # component is not None
            return sum(
                [depth_matches[component] for _, depth_matches in self.matches.items()]
            ) / len(self.matches)

    @abstractmethod
    def error_types(self) -> pd.DataFrame:
        """Returns the errors that occurred in the prediction based on the given reference."""
        pass
