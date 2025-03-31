from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    base_path: str = ""
    recommenders_base_path: str = f"{base_path}/recommendations"


settings = CommonSettings()
