from typing import List


def simplify(wikidata_point):
    """
    Simplify the schema of a Wikidata point keeping only the most important:
    * table id
    * page title
    * section title
    * caption
    * columns
    * rows
    """
    return {
        "table_id": wikidata_point["tableId"],
        "page_title": wikidata_point["pgTitle"],
        "section_title": wikidata_point["sectionTitle"],
        "caption": wikidata_point["tableCaption"],
        "columns": extract_table_columns(wikidata_point["tableHeaders"]),
        "rows": extract_table_rows(wikidata_point["tableData"]),
    }


def extract_table_columns(wikidata_point_headers) -> List[str]:
    """
    Extract the columns of a Wikidata point.
    """
    return [header["text"] for header in wikidata_point_headers[0]]


def extract_table_rows(wikidata_point_rows) -> List[List[str]]:
    """
    Extract the columns of a Wikidata point.
    """
    return [[value["text"] for value in row] for row in wikidata_point_rows]
