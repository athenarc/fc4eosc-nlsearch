from typing import List, Literal

from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    type: str
    driver: str
    username: str
    password: str
    hostname: str
    port: int
    name: str

    id: str
    aliases: list
    schemas: list

    blacklist_tables: list

    test_table: str
    test_hostname: str


class Cordis(DatabaseConfig):
    type: str = "mysql"
    driver: str = "pymysql"
    username: str = ""
    password: str = ""
    hostname: str = "mysql"
    port: int = 3306
    name: str = "cordis"

    id: str = "cordis"
    aliases: list = ["cordis"]
    schemas: list = ["cordis"]

    blacklist_tables: list = []

    test_table: str = "projects"
    test_hostname: str = "darelab.athenarc.gr"

    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_CORDIS_"
        extra = "ignore"


class Faircore(DatabaseConfig):
    type: str = "postgresql"
    driver: str = "psycopg2"
    username: str = ""
    password: str = ""
    hostname: str = "train.darelab.athenarc.gr"
    port: int = 5555
    name: str = "fc4eosc"

    id: str = "fc4eosc"
    aliases: list = ["fc4eosc"]
    schemas: list = ["public"]

    blacklist_tables: list = []

    test_table: str = "community"
    test_hostname: str = "train.darelab.athenarc.gr"

    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_FC4EOSC_"
        extra = "ignore"


class Nestle(DatabaseConfig):
    type: str = "postgresql"
    driver: str = "psycopg2"
    username: str = ""
    password: str = ""
    hostname: str = "train.darelab.athenarc.gr"
    port: int = 5555
    name: str = "nestle"

    id: str = "nestle"
    aliases: list = ["nestle"]
    schemas: list = ["public"]

    blacklist_tables: list = []

    test_table: str = "sales"
    test_hostname: str = "train.darelab.athenarc.gr"

    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_NESTLE_"
        extra = "ignore"


class Datagems(DatabaseConfig):
    type: str = "postgresql"
    driver: str = "psycopg2"
    username: str = ""
    password: str = ""
    hostname: str = "train.darelab.athenarc.gr"
    port: int = 5555
    name: str = "datagems"

    id: str = "datagems"
    aliases: list = ["datagems"]
    schemas: list = ["mathe"]

    blacklist_tables: list = []

    test_table: str = "assessment"
    test_hostname: str = "train.darelab.athenarc.gr"

    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_DATAGEMS_"
        extra = "ignore"


class RDGraph(DatabaseConfig):
    type: str = "postgresql"
    driver: str = "psycopg2"
    username: str = ""
    password: str = ""
    hostname: str = "rdgraph-postgres-beta.openaire.eu"
    port: int = 5432
    name: str = "rdgraph"

    id: str = "rdgraph"
    aliases: list = ["rdgraph"]
    schemas: list = ["public"]

    blacklist_tables: list = []

    test_table: str = "community"
    test_hostname: str = "rdgraph-postgres-beta.openaire.eu"

    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_RDGRAPH_"
        extra = "ignore"


class AvailableDatabases(BaseSettings):
    databases: List[DatabaseConfig] = [
        Cordis(),
        Faircore(),
        RDGraph(),
        Nestle(),
        Datagems(),
    ]


settings = AvailableDatabases()


def get_available_databases():
    dbs = tuple([db.id for db in settings.databases])
    return Literal[dbs]  # type: ignore
