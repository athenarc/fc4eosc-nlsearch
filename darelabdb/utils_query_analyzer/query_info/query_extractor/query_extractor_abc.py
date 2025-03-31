from abc import abstractmethod

from darelabdb.utils_query_analyzer.query_info.query_info import QueryInfo


class QueryExtractor:
    @abstractmethod
    def extract(self, query: str) -> QueryInfo:
        pass
