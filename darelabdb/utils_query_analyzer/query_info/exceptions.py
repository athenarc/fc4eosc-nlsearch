class SQLParserException(Exception):
    """Raised when there is an error in parsing a SQL query with a selected parser"""

    def __init__(self, message):
        super().__init__(message)
