from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, field_validator, model_validator


class ExperimentConfigModel(BaseModel):
    name: str  # Every model available in the ollama API
    options: Optional[
        dict
    ]  # In the options can be added all the options available in the ollama API
    mode: Literal["generate", "chat"]


class ExperimentConfigSimilarExamples(BaseModel):
    selector: str  # The name of the class responsible for the selection of the similar examples.
    num: int  # The number of similar examples that will be present in the prompt.
    examples_pool: list  # A list of the datasets that wil be used as the pool to select similar examples.
    # A certain split of a dataset can be specified by <dataset_name>.<split_name>
    embedding_method: (
        str  # The method that will be used for the creation of the text embeddings.
    )
    only_in_domain: (
        bool  # If True the selected examples must be from the same database.
    )
    representation: Literal[
        "with_schema", "without_schema"
    ]  # A parameter that determines whether the database schema
    # of the similar examples should be represented in the prompt

    @field_validator("num")
    @classmethod
    def num_check(cls, num: int) -> int:
        if num < 0:
            raise ValueError(
                "The number of similar examples must be greater of equal to zero!"
            )
        return num


class ExperimentDbRepresentationExampleValues(BaseModel):
    num: int
    categorical_threshold: Optional[int] = None

    @model_validator(mode="after")
    def categorical_threshold_check(self):
        if self.categorical_threshold is None and self.num != 0:
            raise ValueError(
                "The categorical threshold must be given when the number of sample examples is not zero!"
            )
        elif (
            self.categorical_threshold is not None
            and self.categorical_threshold < self.num
        ):
            logger.warning(
                "The categorical threshold for the values of a column has been set to "
                "less that the number of example values"
            )
        return self

    @field_validator("num")
    @classmethod
    def num_check(cls, num: int) -> int:
        if num < 0:
            raise ValueError(
                "The number of column examples must be greater of equal to zero!"
            )
        return num


class ExperimentConfigDbRepresentation(BaseModel):
    format: Literal["ddl", "compact", "m-schema"]  # The format of the database schema.
    example_values: ExperimentDbRepresentationExampleValues
    include_primary_keys: (
        bool  # If True, the schema will contain information about the primary keys.
    )
    include_foreign_keys: (
        bool  # If True, the schema will contain information about the foreign keys.
    )
    include_notes: bool  # If True, the schema will contain notes about the schema elements (e.g., column descriptions).


class ExperimentConfigPrompt(BaseModel):
    version: str  # The version of the prompt that will be used in the experiment.
    similar_examples: Optional[ExperimentConfigSimilarExamples] = None
    database_representation: ExperimentConfigDbRepresentation

    @model_validator(mode="after")
    def similar_examples_check(self):
        if self.version != "zero_shot_prompt" and self.similar_examples is None:
            raise ValueError(
                "Similar examples must be provided for the selected prompt version!"
            )
        return self


class ExperimentLogging(BaseModel):
    enable_wandb: bool  # If True, the experiment will be logged in wandb.
    project_name: Optional[str] = None  # The name of the project in wandb.
    group_name: Optional[str] = None
    experiment_name: str  # The name of the experiment.
    save_local: bool  # If True, the results will be saved as an Excel file in the specified `save_dir`.
    save_dir: Optional[
        str
    ] = None  # The directory to save the results. If None, the results will be
    # saved in the working directory at the time of execution.

    @model_validator(mode="after")
    def project_name_check(self):
        if self.enable_wandb and self.project_name is None:
            raise ValueError(
                "Project name must be provided to log the experiment in the wandb!"
            )
        return self


class ExperimentConfig(BaseModel):
    """
    The configuration of a text-to-SQL experiment.
    """

    model: ExperimentConfigModel
    prompt: ExperimentConfigPrompt
    evaluated_dataset: str  # The dataset that will be used to run the experiment.
    experiment_logging: ExperimentLogging
    download_dir: Optional[str] = None
