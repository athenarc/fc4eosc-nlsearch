from pydantic_settings import BaseSettings
import os


class CommonSettings(BaseSettings):
    base_path: str = "/api/datagent"


class TestServerSettings(CommonSettings):  # for the deployment in the Test Server
    # spring_service: str = "http://spring-api:8080"
    query_builder_service: str = "https://darelab.athenarc.gr"
    eqsplain_service: str = "https://darelab.athenarc.gr"
    qr2t_service: str = "https://darelab.athenarc.gr"


class ProdSettings(CommonSettings):  # for the deployment in the Production Server
    # spring_service: str = "https://test.darelab.athenarc.gr"
    query_builder_service: str = "http://query-builder"
    eqsplain_service: str = "http://eqsplain"
    qr2t_service: str = "https://darelab.athenarc.gr"


class DevSettings(CommonSettings):  # for a development/test environment
    # spring_service: str = "https://test.darelab.athenarc.gr"
    query_builder_service: str = "https://darelab.athenarc.gr"
    eqsplain_service: str = "https://darelab.athenarc.gr"
    qr2t_service: str = "https://darelab.athenarc.gr"


if "DEV" in os.environ:
    settings = DevSettings()
elif "TEST" in os.environ:  # pragma: no cover
    settings = DevSettings()
elif "TESTSERVER" in os.environ:  # pragma: no cover
    settings = TestServerSettings()
else:
    settings = ProdSettings()  # pragma: no cover
