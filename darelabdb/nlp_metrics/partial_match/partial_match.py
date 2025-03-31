from typing import Union

import mo_parsing
from torchmetrics import Metric
from tqdm import tqdm
from loguru import logger

from darelabdb.nlp_metrics.partial_match.components.operator_match import OperatorsMatch
from darelabdb.nlp_metrics.partial_match.components.structural_match import (
    StructuralMatch,
)
from darelabdb.nlp_metrics.partial_match.components.variable_match import VariablesMatch
from darelabdb.utils_query_analyzer.query_info.query_extractor.mo_sql_parser_extractor import (
    MoQueryExtractor,
)
from darelabdb.utils_query_analyzer.query_info.query_info import QueryInfo


class PartialMatch:
    def __init__(
        self, reference: QueryInfo, prediction: QueryInfo, with_values: bool = True
    ):
        """Returns the match for every component of the partial match"""

        # Remove all 'on' conditions of join and add them in the where clause for better comparison
        reference.move_join_on_condition_in_where()
        prediction.move_join_on_condition_in_where()

        self.structural_match = StructuralMatch(reference, prediction)
        self.operators_match = OperatorsMatch(reference, prediction)
        self.variables_match = VariablesMatch(reference, prediction, with_values)

    @classmethod
    def name(cls):
        return "PM"

    @classmethod
    def components(cls):
        return [StructuralMatch, OperatorsMatch, VariablesMatch]

    @classmethod
    def reported_values(cls, extensive: bool = True):
        if extensive is False:
            return ["PM", "SM", "OM", "VM"]
        else:
            return (
                ["PM", "SM", "OM", "VM"]
                + StructuralMatch.components()
                + OperatorsMatch.components()
                + VariablesMatch.components()
            )

    def value(
        self, depth: int = None, extensive_report: bool = False
    ) -> dict[str, float]:
        report = {
            "PM": (
                self.structural_match.value(depth=depth)
                + self.operators_match.value(depth=depth)
                + self.variables_match.value(depth=depth)
            )
            / 3,
            "SM": self.structural_match.value(depth),
            "OM": self.operators_match.value(depth),
            "VM": self.variables_match.value(depth),
        }

        if extensive_report:
            report.update(
                {
                    component: self.structural_match.value(component=component)
                    for component in StructuralMatch.components()
                }
            )
            report.update(
                {
                    component: self.operators_match.value(component=component)
                    for component in StructuralMatch.components()
                }
            )
            report.update(
                {
                    component: self.variables_match.value(component=component)
                    for component in StructuralMatch.components()
                }
            )
        return report


class PartialMatchMetric(Metric):
    def __init__(self):
        super().__init__()

        self.partial_match_per_pair = {
            m: [] for m in PartialMatch.reported_values(extensive=True)
        }

    @classmethod
    def name(cls) -> str:
        return "partial_match"

    @classmethod
    def reported_values(cls):
        return PartialMatch.reported_values(extensive=True)

    def _compute_aggregated_results(self) -> dict:
        return {
            metric: sum(values) / len(values)
            for metric, values in self.partial_match_per_pair.items()
        }

    def update(self, preds: list[str], targets: list[str], db_paths: list[str]) -> None:
        """
        Updates the values of the partial match results with the calculations for the given predictions and targets.
        Args:
            preds (list[str]): A list with the sql queries predicted
            targets (list[str]): A list with the gold sql queries
            db_paths (list[str]): A list with the sqlite db paths of each prediction-target pair
        """

        # Initialize sql parser
        mo_sql_extractor = MoQueryExtractor()

        for prediction, reference in tqdm(
            zip(preds, targets), desc="Calculating partial match..."
        ):
            # Parse the predicted and the referenced query
            try:
                reference_query_info = mo_sql_extractor.extract(reference)
            except Exception as e:
                logger.warning(
                    "There was an error in parsing the referenced query. The partial match value will be "
                    "set to 0\n"
                    f"SQL query: {reference}\n"
                    f"Error: {e}"
                )
                # Add zero value to all metrics for this pair
                for metric in self.partial_match_per_pair.keys():
                    self.partial_match_per_pair[metric].append(0)
                continue

            try:
                prediction_query_info = mo_sql_extractor.extract(prediction)
            except Exception as e:
                logger.warning(
                    "There was an error in parsing the predicted query. The partial match value will be "
                    "set to 0\n"
                    f"SQL query: {prediction}\n"
                    f"Error: {e}"
                )
                # Add zero value to all metrics for this pair
                for metric in self.partial_match_per_pair.keys():
                    self.partial_match_per_pair[metric].append(0)
                continue

            try:
                # Calculate the partial match
                partial_match = PartialMatch(
                    reference=reference_query_info,
                    prediction=prediction_query_info,
                    with_values=True,
                )

                # For each metric calculated in the partial match component
                for metric, value in partial_match.value().items():
                    self.partial_match_per_pair[metric].append(value)

            except Exception as e:
                logger.warning(
                    "There was an error in calculating partial match. The partial match values will be"
                    "set to 0.\n"
                    f"Error: {e}"
                )
                # Add zero value to all metrics for this pair
                for metric in self.partial_match_per_pair.keys():
                    self.partial_match_per_pair[metric].append(0)

    def compute(self, aggregated: bool = True) -> Union[list, dict]:
        """
        Returns the partial match reported values for the whole corpus (aggregated or per pair).
        Args:
            aggregated (bool): If True, the average structural, operator, variable and sum of the three scores are
                returned for the corpus. Otherwise, a dictionary is returned with the partial match of each pair.
        """
        # If results need to be reported for each pair
        if not aggregated:
            return self.partial_match_per_pair
        else:
            return self._compute_aggregated_results()
