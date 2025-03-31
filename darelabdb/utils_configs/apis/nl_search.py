from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    base_path: str = "/nl_search/api"


settings = CommonSettings()
