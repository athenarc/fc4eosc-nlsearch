from typing import Union

from darelabdb.nlp_metrics.utils.execution_accuracy_calculator import exec_evaluator
from loguru import logger
from torchmetrics import Metric
from tqdm import tqdm


class ExecutionAccuracy(Metric):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.execution_accuracy_results_per_pair = []

    @classmethod
    def name(cls) -> str:
        return "exec"

    def _compute_aggregated_results(
        self, report_only_exec: bool = True
    ) -> Union[dict[str, float], float]:
        if report_only_exec:
            results = sum(
                evaluation["exec"]
                for evaluation in self.execution_accuracy_results_per_pair
            ) / len(self.execution_accuracy_results_per_pair)
        else:
            results = {}
            execution_accuracy_types = list(
                self.execution_accuracy_results_per_pair[0].keys()
            )
            for execution_accuracy_type in execution_accuracy_types:
                if execution_accuracy_type != "error":
                    results[execution_accuracy_type] = sum(
                        evaluation[execution_accuracy_type]
                        for evaluation in self.execution_accuracy_results_per_pair
                    ) / len(self.execution_accuracy_results_per_pair)
        return results

    def update(
        self,
        preds: list[str],
        targets: list[str],
        db_paths: list[str],
    ) -> None:
        """
        Updates the values of the execution accuracy results with the calculations for the given predictions and
        targets.
        Args:
            preds (list[str]): A list with the sql queries predicted
            targets (list[str]): A list with the gold sql queries
            db_paths (list[str]): A list with the sqlite db paths of each prediction-target pair
        """
        for prediction, target, db_path in tqdm(
            zip(preds, targets, db_paths),
            desc="Calculating execution accuracy...",
            total=len(preds),
        ):
            if prediction is None:
                self.execution_accuracy_results_per_pair.append(
                    {
                        "exec": 0,
                        "exec-only_common_result_columns": 0,
                        "exec-target_result_columns_subset": 0,
                        "exec-target_result_columns_superset": 0,
                        "target_exec_time": None,
                        "pred_exec_time": None,
                        "exec_error": None,
                    }
                )
            else:
                try:
                    results = exec_evaluator(
                        db_name=db_path, pred=prediction, target=target
                    )

                    self.execution_accuracy_results_per_pair.append(
                        {
                            **results,
                            "exec_error": None,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error in execution accuracy calculation: {e}")
                    # TODO get the values dynamically
                    self.execution_accuracy_results_per_pair.append(
                        {
                            "exec": 0,
                            "exec-only_common_result_columns": 0,
                            "exec-target_result_columns_subset": 0,
                            "exec-target_result_columns_superset": 0,
                            "target_exec_time": None,
                            "pred_exec_time": None,
                            "exec_error": str(e),
                        }
                    )

    def compute(
        self, aggregated: bool = True, report_only_exec: bool = True
    ) -> Union[list, dict, float]:
        """
        Returns the execution accuracy for the whole corpus (aggregated or per pair).
        Args:
            aggregated (bool): If True, the average execution accuracy score is returned for all the corpus. Otherwise,
            a dictionary is returned with the execution accuracy results of each pair.
            report_only_exec (bool): If True in the aggregated results only the default execution accuracy is reported.
                Otherwise, the report contains all the available execution accuracy types
                (e.g., execution accuracy with only considering the common result columns)
        """
        if not aggregated:
            return self.execution_accuracy_results_per_pair
        else:
            return self._compute_aggregated_results(report_only_exec=report_only_exec)
