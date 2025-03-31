import json
import os
import re


class ValueLinkingPerformance:
    """
    A class to calculate accuracy metrics for value linking tasks, comparing predicted values
    against ground truth data with optional filtering of numeric and symbol-based entries.

    Attributes:
        ground_truth_file (str): Path to the ground truth JSON file.
        output_dir (str): Directory where output logs and results will be saved.
        ground_truth_data (list): Parsed ground truth data loaded from the JSON file.
    """

    def __init__(self, ground_truth_file, output_dir):
        """
        Initialize the ValueLinkingPerformance calculator.

        Args:
            ground_truth_file (str): Path to the JSON file containing ground truth data.
            output_dir (str): Directory path for saving output files and logs.
        """
        self.ground_truth_file = ground_truth_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ground_truth_data = self._load_json(ground_truth_file)

    @staticmethod
    def _load_json(file_path):
        """Load a JSON file and return its content."""
        with open(file_path, "r") as file:
            return json.load(file)

    @staticmethod
    def remove_numeric_values(gt_list):
        """
        Remove strings from the ground truth list where the value part (table.column.value)
        matches specific conditions (only numbers, certain symbols, or ends with '%').
        """
        filtered_list = []
        for entry in gt_list:
            parts = entry.split(".")
            if len(parts) >= 3:
                value = ".".join(parts[2:])
                if re.fullmatch(r"[0-9\-/\.,:\s]+", value) or value.endswith("%"):
                    continue
            filtered_list.append(entry)
        return filtered_list

    def calculate_accuracy_and_log(
        self, predicted_file, output_file, remove_values=False
    ):
        """
        Calculate strict exact match, precision, and column-level recall metrics between
        predicted and ground truth data, with optional filtering of numeric/symbol-based values.

        Args:
            predicted_file (str): Path to the JSON file containing predicted values.
            output_file (str): Path to save the mismatched records log.
            remove_values (bool, optional): Whether to filter numeric/symbol-based values from
                ground truth before calculation. Defaults to False.

        Returns:
            tuple: Metrics including:
                - set_inclusion_with_filter (float): Exact match score (1 if all ground truth
                  entries are present in predictions; 0 otherwise) after filtering.
                - precision_with_filter (float): Precision (correct predictions / total predictions)
                  after filtering numeric/symbol-based values.
                - column_recall_with_filter (float): Column-level recall (whether all ground truth
                  column prefixes are present) after filtering.
                - set_inclusion_without_filter (float): Exact match score without filtering.
                - precision_without_filter (float): Precision score without filtering.
                - column_recall_without_filter (float): Column-level recall score without filtering.
                - average_recall_with_filter (float): Recall (partial matches allowed)
                  after filtering.
                - average_recall_without_filter (float): Recall without filtering.
        """
        predicted_data = self._load_json(predicted_file)

        if len(predicted_data) != len(self.ground_truth_data):
            raise ValueError(
                "The number of records in predicted and ground truth files do not match."
            )

        total_records = len(self.ground_truth_data)
        set_inlusion_with_filter = 0
        set_inclusion_without_filter = 0
        mismatches = []
        total_predicted = 0
        correct_predicted_with_filter = 0
        correct_predicted_without_filter = 0
        column_correct_with_filter = 0
        column_correct_without_filter = 0
        sum_recall_with_filter = 0.0
        sum_recall_without_filter = 0.0

        for index, (pred, gt) in enumerate(zip(predicted_data, self.ground_truth_data)):
            total_predicted += len(pred)

            if remove_values:
                gt_filtered = self.remove_numeric_values(gt)
            else:
                gt_filtered = gt

            if remove_values and not gt_filtered:
                total_records -= 1
                continue

            # Compute prefix sets
            gt_columns_with_filter = set(
                [".".join(value.split(".")[:2]) for value in gt_filtered]
            )
            gt_columns_without_filter = set(
                [".".join(value.split(".")[:2]) for value in gt]
            )
            pred_columns = set([".".join(value.split(".")[:2]) for value in pred])

            # Exact Match-Based Recall
            if all(gt_value in pred for gt_value in gt_filtered):
                set_inlusion_with_filter += 1
            if all(gt_value in pred for gt_value in gt):
                set_inclusion_without_filter += 1

            # Partial Recall Calculation (per-record)
            correct_values_with_filter = sum(
                1 for gt_value in gt_filtered if gt_value in pred
            )
            recall_with_filter = (
                correct_values_with_filter / len(gt_filtered) if gt_filtered else 0
            )
            sum_recall_with_filter += recall_with_filter

            correct_values_without_filter = sum(
                1 for gt_value in gt if gt_value in pred
            )
            recall_without_filter = correct_values_without_filter / len(gt) if gt else 0
            sum_recall_without_filter += recall_without_filter

            # Track correct predicted values for precision
            correct_predicted_with_filter += correct_values_with_filter
            correct_predicted_without_filter += correct_values_without_filter

            # Column Accuracy (prefix-based)
            if gt_columns_with_filter.issubset(pred_columns):
                column_correct_with_filter += 1
            if gt_columns_without_filter.issubset(pred_columns):
                column_correct_without_filter += 1

            # Log mismatches
            if (remove_values and recall_with_filter < 1.0) or (
                not remove_values and recall_without_filter < 1.0
            ):
                mismatches.append(
                    {
                        "index": index,
                        "ground_truth": gt_filtered if remove_values else gt,
                        "predicted": pred,
                        "partial_recall": (
                            recall_with_filter
                            if remove_values
                            else recall_without_filter
                        ),
                    }
                )

        # Calculate Final Metrics
        average_set_inlusion_with_filter = (
            set_inlusion_with_filter / total_records if total_records > 0 else 0
        )
        precision_with_filter = (
            correct_predicted_with_filter / total_predicted
            if total_predicted > 0
            else 0
        )
        column_recall_with_filter = (
            column_correct_with_filter / total_records if total_records > 0 else 0
        )

        average_set_inclusion_without_filter = (
            set_inclusion_without_filter / total_records if total_records > 0 else 0
        )
        precision_without_filter = (
            correct_predicted_without_filter / total_predicted
            if total_predicted > 0
            else 0
        )
        column_recall_without_filter = (
            column_correct_without_filter / total_records if total_records > 0 else 0
        )

        # Average Partial Recall (Macro-Average)
        average_recall_with_filter = (
            sum_recall_with_filter / total_records if total_records > 0 else 0
        )
        average_recall_without_filter = (
            sum_recall_without_filter / total_records if total_records > 0 else 0
        )

        # Log mismatches
        with open(output_file, "w") as out_file:
            json.dump(mismatches, out_file, indent=4)

        return (
            average_set_inlusion_with_filter,
            precision_with_filter,
            column_recall_with_filter,
            average_set_inclusion_without_filter,
            precision_without_filter,
            column_recall_without_filter,
            average_recall_with_filter,
            average_recall_without_filter,
        )


########## Example Usage ##########

"""

if __name__ == "__main__":
    ground_truth_file = "/data/hdd1/users/akouk/BIRD-dev/dev_20240627/dev_strings_lower.json"
    predicted_files = [
        "/data/hdd1/users/akouk/hybrid/final_results/BM25_space.json",
        "/data/hdd1/users/akouk/hybrid/final_results/bm25_values.json",
        "/data/hdd1/users/akouk/hybrid/final_results/BM25FaissFlatMinhashPicklist_space.json",
        "/data/hdd1/users/akouk/hybrid/final_results/BM25FaissFlatMinhashPicklist_values.json",
        "/data/hdd1/users/akouk/hybrid/final_results/CHESS.json",
        "/data/hdd1/users/akouk/hybrid/final_results/CodeS.json",
    ]
    output_dir = "./results/"

    calculator = AccuracyCalculator(ground_truth_file, output_dir)
    table_headers = [
        "File Name", "Accuracy", "Precision", "Column Accuracy",
        "Accuracy (unfiltered)", "Precision (unfiltered)", "Column Accuracy (unfiltered)"
    ]

    results = []
    for predicted_file in predicted_files:
        file_name = os.path.basename(predicted_file).replace('.json', '')
        output_file_remove_values = os.path.join(output_dir, f"{file_name}_no_values.json")

        (
            acc_with_filter, prec_with_filter, prefix_acc_with_filter,
            acc_without_filter, prec_without_filter, prefix_acc_without_filter
        ) = calculator.calculate_accuracy_and_log(
            predicted_file, output_file_remove_values, remove_values=True
        )

        results.append([
            file_name,
            f"{acc_with_filter:.4f}", f"{prec_with_filter:.4f}", f"{prefix_acc_with_filter:.4f}",
            f"{acc_without_filter:.4f}", f"{prec_without_filter:.4f}", f"{prefix_acc_without_filter:.4f}"
        ])

    print(tabulate(results, headers=table_headers, tablefmt="fancy_grid"))

"""
