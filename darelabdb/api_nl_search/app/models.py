from pydantic import BaseModel
from typing import List


class QueryResults(BaseModel):
    columns: List[str]
    index: List[int]
    data: List[List]

    class Config:
        json_schema_extra = {
            "example": {
                "columns": [
                    "firstname",
                    "fromorcid",
                    "fullname",
                    "id",
                    "lastname",
                    "orcid",
                ],
                "index": [0, 1, 2, 3],
                "data:": [
                    [
                        "John",
                        "",
                        "John Campbell",
                        "048fdf78266fcbde6f7080bcba7aa4a8",
                        "Campbell",
                        "",
                    ],
                    [
                        "John",
                        "",
                        "John Ludwig",
                        "05f1e4f3c4b8c93ad510f6ceaa7414db",
                        "Ludwig",
                        "",
                    ],
                    [
                        "John",
                        "",
                        "Penwell , John",
                        "0b065841cb135544eed90f62a6da58e3",
                        "Penwell",
                        "",
                    ],
                    [
                        "John",
                        "",
                        "John Sweeney",
                        "0b0c5bcd25b92ba5b4dcf8440c26e294",
                        "Sweeney",
                        "",
                    ],
                ],
            }
        }


class SqlQuery(BaseModel):
    sql_query: str


class NlQuery(BaseModel):
    nl_query: str

    class Config:
        json_schema_extra = {
            "examples": [
                {"nl_query": "Show all info on authors named John"},
                {"nl_query": "How many authors are there?"},
                {"nl_query": "Show the full names of authors with more than 10 publications"},
            ]
        }


class Message(BaseModel):
    message: str
