import os
from typing import Literal

from pydantic_settings import BaseSettings


# class AbstractDatabase(BaseSettings):
#     id: str

#     blacklist_tables: list
#     blacklist_predicate_tables: list
#     whitelist_predicate_columns: list


# class Cordis(AbstractDatabase):
#     id: str = "cordis"

#     blacklist_tables: list = []
#     blacklist_predicate_tables: list = [
#         "project_erc_panels",
#         "project_programmes",
#         "project_subject_areas",
#         "project_member_country",
#     ]
#     whitelist_predicate_columns: list = []


# class SDSS(AbstractDatabase):
#     id: str = "sdss"

#     blacklist_tables: list = ["soda_%", "mangadapall", "translation_%"]
#     blacklist_predicate_tables: list = [
#         "galspecline",
#         "neighbors",
#         "photoobj",
#         "spplines",
#     ]
#     whitelist_predicate_columns: list = [
#         "specobj.class",
#         "specobj.subclass",
#     ]


class CommonSettings(BaseSettings):
    base_path: str = "/query-builder"

    recommend_tables: str = "/recommend/tables/"
    recommend_attributes: str = "/recommend/attributes/"
    recommend_predicates: str = "/recommend/predicates/"
    recommend_adjust_predicates: str = "/recommend/adjust_predicates/"
    recommend_query: str = "/recommend/query/"
    recommend_queries: str = "/recommend/queries/"

    update: str = "/update/"

    clauses: tuple = ("select", "from", "where", "predicate")

    ucb_alpha: float = 0.1
    pucb_alpha: float = 0.1
    pucb_p: float = 0.5

    not_categorical: tuple = ("date", "boolean", "real")
    categorical_upper_threshold: int = 400
    categorical_lower_threshold: int = 3

    idrecs_file: str = "data/idrecs.json"


class DockerSettings(CommonSettings):
    redis: str = "redis://redis-mabrecs"
    production_dir: str = "data/"


class DevSettings(CommonSettings):
    redis: str = "redis://localhost"
    production_dir: str = "projects/api_query_builder/data/production/"


class TestSettings(CommonSettings):
    redis: str = "redis://redis"
    production_dir: str = "projects/api_query_builder/data/production/"


if "DEV" in os.environ:
    settings = DevSettings()
elif "TEST" in os.environ:
    settings = TestSettings()
else:
    settings = DockerSettings()  # pragma: no cover


def get_available_databases():
    return Literal["cordis"]  # type: ignore


def get_available_clauses():
    return Literal[settings.clauses]  # type: ignore
