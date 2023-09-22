from model.model import generate_outputs, load_model, remove_db_ids
from utils.database import executeSQL
import functools
import os
import sys
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.append(os.path.abspath('../'))

FC4EOSC_SCHEMA = "| research | author: firstname, fromorcid, fullname, id, lastname, orcid | result_author: author_id, result_id | community: id, name, acronym, description | result: accessright, country, description, id, keywords, language, publication_date, publisher, title, type | result_community: community_id, result_id | result_citations: id, result_id_cited, result_id_cites | result_pid: id, pid, result_id, type"

# Load model
MODEL_PATH = "../trained-models/tscholak/1zha5ono"
# MODEL_PATH = "../trained-models/lm100k-text-to-sql/"
model, tokenizer = load_model(MODEL_PATH)


@functools.lru_cache
def cached_generate(model, tokenizer, input_text, prefix, k=4, num_beams=4, do_sample=False):
    return generate_outputs(model, tokenizer, input_text, prefix, k=k, do_sample=do_sample, num_beams=num_beams)


tags_metadata = [
    {
        "name": "text-to-sql",
        "description": "Generate queries from natural language."
    }
]

app = FastAPI(
    title="Text-to-SQL",
    description="Generate SQL queries from NL.",
    version="0.1",
    openapi_url="/text-to-sql/openapi.json",
    docs_url="/text-to-sql/docs",
    redoc_url="/text-to-sql/redoc",
    openapi_tags=tags_metadata
)
app.mount("/text-to-sql/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


################################# Text-to-SQL #################################


@app.get("/text-to-sql", response_class=HTMLResponse, include_in_schema=False)
async def nl2sql(request: Request):
    question = ""
    schema = ""
    sql_preds = []
    return templates.TemplateResponse("nl2sql.html", {"request": request, "question": question, "schema": schema, "sql_preds": sql_preds})


@app.post("/text-to-sql", response_class=HTMLResponse, include_in_schema=False)
async def nl2sql_post(request: Request, question: str = Form(...), schema: str = Form(...)):
    model_input = question + " " + schema
    sql_preds = cached_generate(model, tokenizer, model_input, "")
    sql_preds = remove_db_ids(sql_preds)
    # model_input = question + " database schema: " + schema
    # sql_preds = cached_generate(model, tokenizer, model_input, NL2SQL_PREFIX)
    return templates.TemplateResponse("nl2sql.html", {"request": request, "question": question, "schema": schema, "sql_preds": sql_preds})


############################## FC4EOSC NL Search ###############################


@app.get("/text-to-sql/fc4eosc", response_class=HTMLResponse, include_in_schema=False)
async def fc4eosc(request: Request):
    question = ""
    sql_preds = []
    execution_results = []
    results = ""

    return templates.TemplateResponse("fc4eosc.html", {"request": request, "question": question, "sql_preds": sql_preds, "execution_results": execution_results, "results": results})


@app.post("/text-to-sql/fc4eosc", response_class=HTMLResponse, include_in_schema=False)
async def fc4eosc_post(request: Request, question: str = Form(...)):
    model_input = question + " " + FC4EOSC_SCHEMA
    sql_preds = cached_generate(model, tokenizer, model_input, "")
    sql_preds = remove_db_ids(sql_preds)
    execution_results = ["unknown"] * len(sql_preds)

    results = ""
    for i, sql_pred in enumerate(sql_preds):
        try:
            results = executeSQL(sql_pred, 'fc4eosc', limit=10)
        except:
            execution_results[i] = "fail"
            continue
        else:
            execution_results[i] = "success"
            results = results.to_html(
                classes="table table-striped bg-light", justify="left")
            break

    return templates.TemplateResponse("fc4eosc.html", {"request": request, "question": question, "sql_preds": sql_preds, "execution_results": execution_results, "results": results})


################################# Other Pages #################################


@app.get("/text-to-sql/contact", response_class=HTMLResponse, include_in_schema=False)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

##################################### API #####################################


@app.post("/text-to-sql/api/sql_query/", tags=["text-to-sql"])
def get_sql_query(nlQuestion: str, database: str, schema: str, k: int = 1):
    """
    Returns k queries for a given NL Question.

    Arguments:
    - nlQuestion (str): A string containing the NL Question
    - database (str): The name of the database that the query is run on
    - schema (str): The schema of the database in a serialised format
    - k (int): The number of queries to generate

    Returns:
    - queries (list of str): A list containing the generated queries
    """

    model_input = f"{nlQuestion} | {database} | {schema}"
    sql_preds = cached_generate(model, tokenizer, model_input, "", k=k)
    sql_preds = remove_db_ids(sql_preds)
    return {"queries": sql_preds}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
