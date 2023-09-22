import requests
import pandas as pd


def executeSQL(query, database, limit=10):
    url = "http://darelab.athenarc.gr/api/database/sql"
    options = {"database": database, "query": query, "limit": limit}
    try:
        resp = requests.post(url, json=options)
        resp_json = resp.json()
        if "error" in resp_json:
            return resp_json["error"]
        return pd.read_json(resp.text, orient="split")

    # TODO: Better exception handling
    except Exception as e:
        raise Exception(f"Query Execution error for {options}")
