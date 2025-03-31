from dataclasses import dataclass
from io import StringIO
from typing import Dict, List

import pandas as pd


@dataclass
class Annotation:
    table_id: str
    query: str
    table_name: str
    query_description: str
    results_description: str
    result: str
    difficulty: int = 1


def create_metadata(annotation: Annotation, use_query: bool) -> str:
    page_title = f"<page_title> {annotation.table_name} </page_title>"
    section_title = (
        f"<section_title> {annotation.query_description if use_query else annotation.table_name} "
        f"</section_title>"
    )

    return page_title + " " + section_title


def create_cell(col_name: str, col_value: str) -> str:
    return f"<cell> {col_value} <col_header> {col_name} </col_header> </cell>"


def extract_cells_from_df(table_df: pd.DataFrame) -> List[Dict[str, str]]:
    cell_list = []
    for _, row in table_df.iterrows():
        for name, val in row.items():
            try:
                added_value = int(val) if int(val) == val else val
            except ValueError:
                added_value = val
            cell_list.append({"col_header": name, "col_val": str(added_value)})

    return cell_list


def parse_str_csv(str_df: str) -> pd.DataFrame:
    data = StringIO(str_df)
    return pd.read_csv(data, sep=",", index_col=0)


def wikisql_datapoint_diff1_to_totto(
    annotation: Annotation, use_query: bool
) -> Dict[str, str]:
    totto_datapoint = {}

    metadata = create_metadata(annotation, use_query)
    table_cells = extract_cells_from_df(parse_str_csv(annotation.result))
    table_cells_str = ""
    for cell_dict in table_cells:
        table_cells_str += (
            create_cell(cell_dict["col_header"], cell_dict["col_val"]) + " "
        )

    totto_datapoint["subtable_and_metadata"] = (
        f"{metadata} <table> {table_cells_str}</table>"
    )
    totto_datapoint["final_sentence"] = annotation.results_description

    return totto_datapoint


def to_totto(datapoints: List[dict], use_query: bool = False) -> List[dict]:
    return [
        wikisql_datapoint_diff1_to_totto(Annotation(**datapoint_dict), use_query)
        for datapoint_dict in datapoints
    ]
