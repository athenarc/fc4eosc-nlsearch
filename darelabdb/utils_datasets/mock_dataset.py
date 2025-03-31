import pandas as pd


def create_mock_datasets(name: str):
    if name == "spider":

        class MockSpider:
            def get(self):
                return {
                    "train": pd.DataFrame(
                        [
                            ["SAMPLE_QUERY 1", "SAMPLE_QUESTION 1", "SAMPLE_DB_ID 1"],
                            ["SAMPLE_QUERY 2", "SAMPLE_QUESTION 2", "SAMPLE_DB_ID 2"],
                        ],
                        columns=["query", "question", "db_id"],
                    ),
                    "dev": pd.DataFrame(
                        [
                            ["SAMPLE_QUERY 1", "SAMPLE_QUESTION 1", "SAMPLE_DB_ID 1"],
                            ["SAMPLE_QUERY 2", "SAMPLE_QUESTION 2", "SAMPLE_DB_ID 2"],
                        ],
                        columns=["query", "question", "db_id"],
                    ),
                }

            def get_schema(self, db_id):
                return {
                    "table_names": [
                        "SAMPLE_TABLE_1",
                        "SAMPLE_TABLE_2",
                        "SAMPLE_TABLE_3",
                    ],
                    "column_names": [
                        [-1, "*"],
                        [0, "SAMPLE_COLUMN_1_1"],
                        [0, "SAMPLE_COLUMN_1_2"],
                        [0, "SAMPLE_COLUMN_1_3"],
                        [1, "SAMPLE_COLUMN_2_1"],
                        [1, "SAMPLE_COLUMN_2_2"],
                        [1, "SAMPLE_COLUMN_2_2"],
                        [2, "SAMPLE_COLUMN_3_1"],
                        [2, "SAMPLE_COLUMN_3_2"],
                        [2, "SAMPLE_COLUMN_3_3"],
                    ],
                    "foreign_keys": [[1, 5], [5, 7]],
                }

            def get_db_path(self, db_id):
                return "SAMPLE_DB_PATH"

        return MockSpider()
