import os
from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    base_path: str = "/api/datadazzle/category-based-recommender"

    get_categories: str = "/categories/"
    recommend: str = "/recommend/"
    update: str = "/update/"

    ucb_alpha: float = 0.1
    pucb_alpha: float = 0.1
    pucb_p: float = 0.5


class DockerSettings(CommonSettings):
    redis: str = "redis://redis-mabrecs/1"
    production_dir: str = "data/datadazzle/"


class DevSettings(CommonSettings):
    redis: str = "redis://localhost/1"
    production_dir: str = "projects/api_faircore_mabrecs/data/datadazzle/"


if "DEV" in os.environ or "TEST" in os.environ:
    settings = DevSettings()
else:
    settings = DockerSettings()  # pragma: no cover
