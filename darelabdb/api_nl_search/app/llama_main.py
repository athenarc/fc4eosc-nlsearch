import json
import os

import pandas as pd
import uvicorn
from darelabdb.api_nl_search.app.database import executeSQL, get_filtered_fc4e_schema
from darelabdb.api_nl_search.app.models import Message, NlQuery, QueryResults, SqlQuery
from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint
from darelabdb.nlp_embeddings.embedding_storage.Pgvector import PgVector
from darelabdb.nlp_text_to_sql.utils.timer import Timer
from darelabdb.nlp_text_to_sql_llm.ollama import handle_text_to_sql
from darelabdb.nlp_text_to_sql_llm.similar_examples_selector.similar_nl_selector import (
    SimilarNLSelector,
)
from darelabdb.utils_query_analyzer.process_sql import get_base_query_limit
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

load_dotenv()

ERROR_MSG = (
    '<div style="margin-bottom: 1em;" class="bg-danger p-3 rounded"> Sorry, could not generate a valid SQL '
    "query for your question </div>"
)

# These columns will be filtered out when loading the database schema
EXCLUDED_COLUMNS = {
    "author": ["id", "firstname", "lastname"],
    "community": ["id"],
    "datasource": ["id"],
    "fos": ["id"],
    "result": ["id"],
    "result_author": ["author_id", "result_id"],
    "result_citations": ["id", "result_id_cited", "result_id_cites"],
    "result_collectedfrom": ["id"],
    "result_community": ["community_id", "result_id"],
    "result_fos": ["fos_id", "result_id"],
    "result_hostedby": ["result_id", "datasource_id"],
}

# Set the schema to None, it will be loaded when needed
FC4E_DB_SCHEMA = None

# Initialize Similar NL Selector for retrieving few-shot examples
similar_nl_selector = SimilarNLSelector(
    embedding_db=PgVector(
        db_name="fc4eosc",
        host="train.darelab.athenarc.gr",
        port="5555",
        user=os.getenv("DATABASE_FC4EOSC_USERNAME"),
        password=os.getenv("DATABASE_FC4EOSC_PASSWORD"),
        primary_key_col_name="nl_question",
        embedding_col_name="embedding",
        schema_name="few_shot_example_embeddings",
        table_name="examples",
        column_types={
            "nl_question": "TEXT",
            "sql_query": "TEXT",
            "embedding": "vector(384)",
        },
    )
)

OLLAMA_CONFIG = {
    "model": {
        "name": "llama3.1:70b",
        "options": {"temperature": 0},
        "mode": "generate",
    },
    "prompt": {
        "version": "few_shot_prompt",
        "similar_examples": {
            "num": 5,
            "only_in_domain": False,
            "representation": "without_schema",
        },
        "database_representation": {
            "format": "ddl",
            "example_values": {"num": 0, "categorical_threshold": None},
            "include_primary_keys": True,
            "include_foreign_keys": True,
            "include_notes": True,
        },
    },
}


def inference(nlQuestion) -> str | None:
    global FC4E_DB_SCHEMA
    if FC4E_DB_SCHEMA is None:
        FC4E_DB_SCHEMA = get_filtered_fc4e_schema(EXCLUDED_COLUMNS)

    with Timer("Generation"):
        # TODO change it
        datapoint = SqlQueryDatapoint(
            nl_query=nlQuestion,
            db_schema=FC4E_DB_SCHEMA,
            sql_query="",
            db_id="",
            db_path="",
        )
        sql_query = handle_text_to_sql(datapoint, OLLAMA_CONFIG, similar_nl_selector)

    return sql_query


def execute(sql, database):
    with Timer("Execution"):
        query_limit = get_base_query_limit(sql)
        results = executeSQL(
            sql,
            database,
            limit=query_limit if query_limit is not None and query_limit < 60 else 60,
        )

    if not isinstance(results, pd.DataFrame):
        # This will happen if error is not caught by sql_is_valid()
        logger.warning(
            f"Database failed to execute SQL: `{sql}`, with error: `{results}`"
        )
        results = None

    return results


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

# NOTE: These checks are put here so that we can run integration tests without path errors
# When deploying the app with docker these directories will be available
if os.path.isdir("static"):
    app.mount("/nl_search/static", StaticFiles(directory="static"), name="static")
if os.path.isdir("templates"):
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
    sql_query = inference(question)
    results = execute(sql_query, "fc4eosc")

    if sql_query is None or results is None:
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
            "predicted_sql": sql_query,
            "sql_candidates": None,
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

    sql_query = inference(user_query.nl_query)

    if sql_query is None:
        return JSONResponse(
            status_code=422,
            content={
                "message": "Sorry, could not generate a valid SQL query for your question"
            },
        )
    else:
        return JSONResponse(content={"sql_query": sql_query})


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
    sql_query = inference(user_query.nl_query)
    logger.info(f"Predicted SQL Query: {sql_query}")

    results = execute(sql_query, "fc4eosc")

    if sql_query is None or results is None:
        return JSONResponse(
            status_code=422,
            content={
                "message": "Sorry, could not generate a valid SQL query for your question"
            },
        )
    else:
        return JSONResponse(content=json.loads(results.to_json(orient="split")))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
