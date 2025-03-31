import json
from typing import Dict, List, Optional

import requests
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_text_to_sql_llm.prompts import (
    FewShotPromptFactory,
    PromptFactory,
    ZeroShotPromptFactory,
)
from darelabdb.nlp_text_to_sql_llm.prompts.prompt_factory_picker import (
    pick_prompt_factory,
)
from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_nl_selector import (
    SimilarNLSelector,
)
from loguru import logger

OLLAMA_BASE_URL = "http://gaia-gpu-2.imsi.athenarc.gr:11434"


def handle_text_to_sql(
    datapoint: SqlQueryDatapoint,
    config: Dict,
    example_retriever: Optional[SimilarNLSelector] = None,
) -> str:
    """
    This is a temporary function to unify different bases that perform Text-to-SQL.

    Args:
        datapoint: The datapoint that contains the natural language question that will be translated and information
                    about the queried database.
        config: Dictionary of configuration parameters. The dictionary must contain:
            - prompt: A dictionary with the prompt information that will be used as input to the model.
            - model: Information about the model used for the translation.
            More info components/darelabdb/nlp_text_to_sql_llm/experiments/example_experiment_config.yml.
        example_retriever: The class responsible for retrieving similar examples.

    Returns:
        The translated SQL query.
    """

    def create_datapoint(row) -> SqlQueryDatapoint:
        return SqlQueryDatapoint(
            nl_query=row["nl_question"],
            sql_query=row["sql_query"],
            db_id=row["db_id"] if "db_id" in row else "",
            db_path="",
        )

    # Get prompt factory
    db_representation_config = config["prompt"]["database_representation"]

    prompt_arguments = {
        "prompt_version": config["prompt"]["version"],
        "schema_type": db_representation_config["format"],
        "include_pk": db_representation_config["include_primary_keys"],
        "include_fk": db_representation_config["include_foreign_keys"],
        "include_notes": db_representation_config["include_notes"],
        "values_num": db_representation_config["example_values"]["num"],
        "categorical_threshold": db_representation_config["example_values"][
            "categorical_threshold"
        ],
    }

    if (
        "similar_examples" in config["prompt"]
        and config["prompt"]["similar_examples"] is not None
    ):
        prompt_arguments["similar_examples_representation"] = config["prompt"][
            "similar_examples"
        ]["representation"]

    prompt_factory = pick_prompt_factory(**prompt_arguments)

    if example_retriever is not None:

        # If we want only examples from the same database
        eq_filters = (
            None
            if not config["prompt"]["similar_examples"]["only_in_domain"]
            else {"db_id": datapoint.db_id}
        )

        # Get few-shot examples
        examples_df = example_retriever.get_similar_examples(
            nl_question=datapoint.nl_query,
            num=config["prompt"]["similar_examples"]["num"],
            eq_filters=eq_filters,
        )

        # Convert examples to datapoints
        examples = [create_datapoint(row) for _, row in examples_df.iterrows()]
    else:
        examples = None

    # Generate SQL query with Ollama
    datapoint = ollama_text_to_sql(
        datapoint,
        config["model"],
        prompt_factory,
        examples=examples,
    )

    return datapoint.prediction


def ollama_text_to_sql(
    datapoint: SqlQueryDatapoint,
    config: dict,
    prompt_factory: PromptFactory,
    examples: Optional[List[SqlQueryDatapoint]] = None,
) -> SqlQueryDatapoint:
    """Uses ollama to perform Text-to-SQL."""

    if issubclass(type(prompt_factory), FewShotPromptFactory) and examples is None:
        raise AttributeError("You have given a few-shot prompt, but no examples!")

    if config["mode"] == "chat":
        if issubclass(type(prompt_factory), FewShotPromptFactory):
            chat = prompt_factory.fill_chat(datapoint, examples)
        else:
            chat = prompt_factory.fill_chat(datapoint)
        datapoint.model_input = chat
        generated_text = chat_generate(chat, config)
    elif config["mode"] == "generate":
        if issubclass(type(prompt_factory), FewShotPromptFactory):
            prompt = prompt_factory.fill_prompt(datapoint, examples)
        else:
            prompt = prompt_factory.fill_prompt(datapoint)
        datapoint.model_input = prompt
        generated_text = generate(prompt, config)
    else:
        raise ValueError(
            "The requested mode is not supported! The supported modes are 'chat' and 'generate'."
        )

    datapoint.model_output = generated_text

    # Get the sql from the response
    if generated_text is not None:
        sql = prompt_factory.extract_sql(generated_text)
    else:
        sql = None
    datapoint.prediction = sql

    return datapoint


def generate(prompt: str, config: Dict) -> Optional[str]:
    """Generate a response to the given prompt with ollama.

    Args:
        prompt (str): The input prompt for the LLM.
        config (Dict): The configuration that contains:
            - NAME: The name of the model used and
            - OPTIONS (Optional): A dictionary with extra options for the model.

    Returns:
        If the operation was successful the generated text is returned, otherwise None.
    """
    request_body = {
        "model": config["name"],
        "prompt": prompt,
        "stream": False,
    }

    # Add extra options (e.g., temperature)
    if "options" in config:
        request_body.update({"options": config["options"]})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", data=json.dumps(request_body)
        )
        response.raise_for_status()
        generated_text = response.json()["response"]
    except requests.exceptions.HTTPError as e:
        logger.error(e)
        generated_text = None

    return generated_text


def chat_generate(chat: List[Dict[str, str]], config: Dict) -> Optional[str]:
    """Generate a response to the given chat with ollama.

    Args:
        chat (List[Dict[str, str]]): The input chat for the LLM.
        config (Dict): The configuration that contains:
            - NAME: The name of the model used and
            - OPTIONS (Optional): A dictionary with extra options for the model.

    Returns:
        If the operation was successful the generated text is returned, otherwise None.
    """
    request_body = {
        "model": config["name"],
        "messages": chat,
        "stream": False,
    }

    # Add extra options (e.g., temperature)
    if "options" in config:
        request_body.update({"options": config["options"]})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", data=json.dumps(request_body)
        )
        response.raise_for_status()
        chat_response = response.json()["message"]
        generated_text = chat_response["content"]
    except requests.exceptions.HTTPError as e:
        logger.error(e)
        generated_text = None

    return generated_text
