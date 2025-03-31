from darelabdb.nlp_data_processing.sql_datapoint_model import SqlQueryDatapoint


def add_sql_whitespaces(datapoint: SqlQueryDatapoint) -> SqlQueryDatapoint:
    """
    Inserts whitespaces before and after each '_', '.' character in column and tables in the given datapoint's sql query
    to improve tokenization with natural language models tokenizers.

    Args:
        datapoint (SqlQueryDatapoint): The datapoint that contains a sql query in which we apply the processing.

    Returns:
        SqlQueryDatapoint: The datapoint with the updated sql query.
    """
    sql_query_words = datapoint.sql_query.split()
    for i, word in enumerate(sql_query_words):
        inside_quotes = False
        if word.startswith('"') or word.startswith("'") or word.startswith('"'):
            inside_quotes = not inside_quotes

        # If this is not a variable in the sql query
        if not inside_quotes:
            for ch in [".", "_"]:
                if ch in word:
                    word = word.replace(ch, f" {ch} ")
            sql_query_words[i] = word

    datapoint.sql_query = " ".join(sql_query_words)
    return datapoint
