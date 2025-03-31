import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

MAPPING_FILE = "mapping.json"
CACHE_DIR = str(Path.home()) + "/.cache/darelabdb/query_results_cache/"


class QueryCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        if not os.path.exists(cache_dir):
            logger.info(f"Cache directory does not exist. Creating at {cache_dir}")
            os.makedirs(cache_dir)

        if not os.path.exists(cache_dir + MAPPING_FILE):
            with open(cache_dir + MAPPING_FILE, "w") as f:
                f.write("{}")

        self.cache_dir = cache_dir

        with open(cache_dir + MAPPING_FILE, "r") as f:
            self.mapping: Dict[str, str] = json.load(f)

    @staticmethod
    def hash_query_value(query: str) -> str:
        hash_object = hashlib.sha256()
        hash_object.update(query.encode("utf-8"))
        return hash_object.hexdigest()

    def exists(self, query: str) -> bool:
        return self.hash_query_value(query) in self.mapping

    def get(self, query: str) -> Optional[pd.DataFrame]:
        if not self.exists(query):
            return None
        results_path = self.mapping[self.hash_query_value(query)]
        return pd.read_csv(results_path)

    def set(self, query: str, results: pd.DataFrame) -> None:
        results_path = self.cache_dir + self.hash_query_value(query) + ".csv"
        results.to_csv(results_path, index=False)
        self.mapping[self.hash_query_value(query)] = results_path
        with open(self.cache_dir + MAPPING_FILE, "w") as f:
            json.dump(self.mapping, f)


def cache_query_results(query_execution_func):
    def wrapper(*args, **kwargs):
        if "sql" not in kwargs:
            logger.error(
                'To enable caching, the "sql" parameter must be passed as a keyword argument.'
            )
        query = kwargs["sql"]

        cache = QueryCache(CACHE_DIR)
        if cache.exists(query):
            logger.info("Query results found in cache. Returning cached results.")
            return cache.get(query)

        results = query_execution_func(*args, **kwargs)

        if not isinstance(results, Dict):
            # Avoid caching error queries since the error might be caused from various reasons (i.e. failing connection,
            # database recovery, etc.)
            cache.set(query, results)

        return results

    return wrapper
