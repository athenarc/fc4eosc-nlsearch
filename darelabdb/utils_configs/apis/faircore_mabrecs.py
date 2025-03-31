import os
from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    base_path: str = "/api/faircore/category-based-recommender"

    get_communities: str = "/available-communities"
    recommend: str = "/recommend/"
    update: str = "/update/"

    ucb_alpha: float = 0.1
    pucb_alpha: float = 0.1
    pucb_p: float = 0.5


class DockerSettings(CommonSettings):
    redis: str = "redis://redis-mabrecs/2"
    production_dir: str = "data/faircore/"


class DevSettings(CommonSettings):
    redis: str = "redis://localhost/2"
    production_dir: str = "projects/api_faircore_mabrecs/data/faircore/"


class TestSettings(CommonSettings):
    redis: str = "redis://redis/2"
    production_dir: str = "projects/api_faircore_mabrecs/data/faircore/"


if "DEV" in os.environ:
    settings = DevSettings()
elif "TEST" in os.environ:
    settings = TestSettings()
else:
    settings = DockerSettings()  # pragma: no cover
