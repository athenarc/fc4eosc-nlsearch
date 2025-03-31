from typing import Callable, Dict, List, Optional, Tuple

from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


class SequentialProcessingExecutor:
    """
    The Sequential Processing executor, creates a pipeline of processing steps where the output of each processing step
        is the input for the next processing step.
    """

    def __init__(
        self,
        input_steps: List[Tuple[Callable, Dict]],
        output_steps: Optional[List[Tuple[Callable, Dict]]] = None,
    ) -> None:
        """
        Initialise the Sequential Processing Executor by providing a list of processing steps to create the `model_input`
            and a dictionary of keyword arguments for each step, as well as an optional list of processing steps and a
            dictionary of keyword arguments to create the `expected_output`.


        !!! warning "processing steps order"

            The input_steps and output_steps are applied sequentially to the **same** datapoint. The order of the processing
            steps in each list is the order of their execution.


        Args:
            input_steps (List[Tuple[Callable, Dict]]): A list of tuples where each tuple contains a callable
                processing step and a dictionary of the step's arguments and the value for each argument (like kwargs).
            output_steps (Optional[List[Tuple[Callable, Dict]]]): An optional list of tuples where each
                tuple contains a callable processing step and a dictionary of the step's arguments and the value for
                each argument (like kwargs). Defaults to None.
        """
        self.input_steps = input_steps
        self.output_steps = output_steps

    def process(self, data_point: SqlQueryDatapoint) -> SqlQueryDatapoint:
        """
        Apply the processing steps in the given datapoint.

        Args:
            data_point (SqlQueryDatapoint): The item to be processed by the executor

        Returns:
            SqlQueryDatapoint: The processed item.
        """
        # Prepare model input
        for step, kwargs in self.input_steps:
            data_point = step(data_point, **kwargs)

        # Prepare expected output, if output steps are given
        if self.output_steps:
            for step, kwargs in self.output_steps:
                data_point = step(data_point, **kwargs)

        return data_point
