import functools
import json
import logging
from enum import Enum

import pandas as pd
import uvicorn
from darelabdb.api_nl_search.app.database import executeSQL
from darelabdb.api_nl_search.app.models import Message, NlQuery, QueryResults, SqlQuery
from darelabdb.nlp_data_processing.processing_steps import (
    remove_db_id,
    remove_extra_whitespace,
    serialize_schema,
    sql_keep_executable,
    sql_transpile,
    text_assemble,
)
from darelabdb.nlp_data_processing.sequential_processing_executor import (
    SequentialProcessingExecutor,
)
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_experiment_runner.inference import inference
from darelabdb.nlp_text_to_sql.model import Seq2SeqText2SqlModel
from darelabdb.nlp_text_to_sql.utils.sql import sql_is_valid
from darelabdb.nlp_text_to_sql.utils.timer import Timer
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ERROR_MSG = '<div style="margin-bottom: 1em;" class="bg-danger p-3 rounded"> Sorry, could not generate a valid SQL query for your question </div>'


class SupportedDB(str, Enum):
    fc4osc = "fc4eosc"
    cordis = "cordis"


# Create processing pipelines
preprocessor = SequentialProcessingExecutor(
    input_steps=[
        (
            remove_extra_whitespace,
            {"source_field": "nl_query", "dest_field": "nl_query"},
        ),
        (
            serialize_schema,
            {"with_db_id": True},
        ),
        (
            text_assemble,
            {
                "recipe": "{{datapoint.nl_query}} | {{datapoint.serialized_schema}}",
                "dest_field": "model_input",
            },
        ),
    ]
)
postprocessor = SequentialProcessingExecutor(
    [
        (remove_db_id, {"separator": "|", "expect_candidates": True}),
        (
            sql_transpile,
            {
                "read_dialect": "mysql",
                "write_dialect": "postgres",
                "expect_candidates": True,
            },
        ),
        (sql_keep_executable, {"dialect": "postgres"}),
    ]
)

# Load model
MODEL_PATH = "../trained-models/tscholak/1zha5ono"
model = Seq2SeqText2SqlModel(
    MODEL_PATH, postprocessor=postprocessor, num_return_sequences=4
)

with open("fc4eosc_schema.json", "r") as fp:
    fc4eosc_schema = json.load(fp)


@functools.lru_cache
def cached_inference(nlQuestion, database):
    if database == "fc4eosc":
        datapoint = SqlQueryDatapoint(
            nl_query=nlQuestion,
            db_id="research",
            db_schema=fc4eosc_schema,
            db_path="",
            sql_query="",
        )

    return inference(model, preprocessor, datapoint)


def inference_and_execute(nlQuestion, database):
    with Timer("Generation"):
        datapoint = cached_inference(nlQuestion, database)

    candidates = datapoint.candidates
    prediction = datapoint.prediction

    if prediction is None:
        results = None
    else:
        with Timer("Execution"):
            results = executeSQL(prediction, "fc4eosc", limit=10)

        if not isinstance(results, pd.DataFrame):
            # This will happen if error is not caught by sql_is_valid()
            logging.warning(
                f"Database failed to execute SQL: `{prediction}`, with error: `{results}`"
            )
            prediction = None
            results = None

    return results, prediction, candidates


tags_metadata = [
    {
        "name": "fc4e",
        "description": "NL Search service for the FAIRCORE4EOSC project",
    },
    {
        "name": "text-to-sql",
        "description": "General text-to-sql API calls",
    },
]

app = FastAPI(
    title="Text-to-SQL",
    description="Talk with your database in natural language",
    version="0.1",
    openapi_url="/nl_search/openapi.json",
    docs_url="/nl_search/docs",
    redoc_url="/nl_search/redoc",
    openapi_tags=tags_metadata,
)
app.mount("/nl_search/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Add CORSMiddleware to avoid CORS errors to the RDGraph Portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


################################# Text-to-SQL #################################


# TODO: Enable Text-to-SQL with custom schema
@app.get("/nl_search", response_class=HTMLResponse, include_in_schema=False)
async def nl2sql():
    response = RedirectResponse(url="/nl_search/fc4eosc")
    return response


# async def nl2sql(request: Request):
#     question = ""
#     schema = ""
#     sql_preds = []
#     return templates.TemplateResponse(
#         "nl2sql.html",
#         {
#             "request": request,
#             "question": question,
#             "schema": schema,
#             "sql_preds": sql_preds,
#         },
#     )


# @app.post("/nl_search", response_class=HTMLResponse, include_in_schema=False)
# async def nl2sql_post(
#     request: Request, question: str = Form(...), schema: str = Form(...)
# ):
#     model_input = question + " " + schema
#     sql_preds = cached_generate(model, tokenizer, model_input, "")
#     sql_preds = remove_db_ids(sql_preds)
#     # model_input = question + " database schema: " + schema
#     # sql_preds = cached_generate(model, tokenizer, model_input, NL2SQL_PREFIX)
#     return templates.TemplateResponse(
#         "nl2sql.html",
#         {
#             "request": request,
#             "question": question,
#             "schema": schema,
#             "sql_preds": sql_preds,
#         },
#     )


############################## FC4EOSC NL Search ###############################


@app.get("/nl_search/fc4eosc", response_class=HTMLResponse, include_in_schema=False)
async def fc4eosc(request: Request):
    question = ""
    predicted_sql = None
    sql_candidates = None
    results = ""

    return templates.TemplateResponse(
        "fc4eosc.html",
        {
            "request": request,
            "question": question,
            "predicted_sql": predicted_sql,
            "sql_candidates": sql_candidates,
            "results": results,
        },
    )


@app.post("/nl_search/fc4eosc", response_class=HTMLResponse, include_in_schema=False)
async def fc4eosc_post(request: Request, question: str = Form(...)):
    results, prediction, candidates = inference_and_execute(question, "fc4eosc")

    if prediction is None:
        results = ERROR_MSG
    else:
        results = results.to_html(
            classes="table table-striped bg-light",
            justify="left",
            index=True if len(results.index) > 1 else False,
        )

    return templates.TemplateResponse(
        "fc4eosc.html",
        {
            "request": request,
            "question": question,
            "predicted_sql": prediction,
            "sql_candidates": candidates,
            "results": results,
        },
    )


################################# Other Pages #################################


@app.get("/nl_search/contact", response_class=HTMLResponse, include_in_schema=False)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


##################################### API #####################################


@app.post(
    "/nl_search/api/fc4e_get_sql/",
    response_model=SqlQuery,
    responses={422: {"model": Message}},
    tags=["fc4e"],
)
def get_sql_query(user_query: NlQuery):
    """
    Returns the predicted SQL query for a given NL Question that is posed to the
        FAIRCORE4EOSC project's database.
    """

    datapoint = cached_inference(user_query.nl_query, "fc4eosc")
    prediction = datapoint.prediction

    if prediction is None:
        return JSONResponse(
            status_code=422,
            content={
                "message": "Sorry, could not generate a valid SQL query for your question"
            },
        )
    else:
        return JSONResponse(content={"sql_query": prediction})


@app.post(
    "/nl_search/api/fc4e_get_results/",
    response_model=QueryResults,
    responses={422: {"model": Message}},
    tags=["fc4e"],
)
def fc4e_get_results(user_query: NlQuery):
    """
    Returns the results for a given NL Question that is posed to the FAIRCORE4EOSC
        project's database.
    """

    results, prediction, candidates = inference_and_execute(
        user_query.nl_query, "fc4eosc"
    )

    if prediction is None:
        return JSONResponse(
            status_code=422,
            content={
                "message": "Sorry, could not generate a valid SQL query for your question"
            },
        )
    else:
        return JSONResponse(content=json.loads(results.to_json(orient="split")))


# TODO: Enable Text-to-SQL with custom schema
# @app.post("/nl_search/api/custom_db_sql_query/", tags=["text-to-sql"])
# def custom_db_get_sql_query(nlQuestion: str, database: str, schema: str, k: int = 1):
#     """
#     Returns k queries for a given NL Question for any database, but the schema
#     must be provided by the user.

#     Arguments:
#     - nlQuestion (str): A string containing the NL Question
#     - database (str): The name of the database that the query is run on
#     - schema (str): The schema of the database in a serialised format
#     - k (int): The number of queries to generate

#     Returns:
#     - queries (list of str): A list containing the generated queries
#     """

#     model_input = f"{nlQuestion} | {database} | {schema}"
#     sql_preds = cached_generate(model, tokenizer, model_input, "", k=k)
#     sql_preds = remove_db_ids(sql_preds)
#     return {"queries": sql_preds}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
