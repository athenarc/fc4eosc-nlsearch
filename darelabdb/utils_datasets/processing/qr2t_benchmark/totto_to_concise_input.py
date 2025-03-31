import re

from dateutil.parser import parse


def is_numeric(cell: str) -> bool:
    try:
        _ = float(cell.replace(",", ""))
        return True
    except ValueError:
        pass

    try:
        _ = float(cell.replace(" ", "").replace("-", ""))
        return True
    except ValueError:
        pass

    return False


def is_date(cell: str) -> bool:
    try:
        _ = parse(cell, fuzzy=False)
        return True
    except (ValueError, OverflowError):
        return False


def find_table(datapoint):
    reg = r"<page_title> (.*) </page_title>"
    return re.search(reg, datapoint)[1]


def find_section(datapoint):
    reg = r"<section_title> (.*) </section_title>"

    try:
        found_section = re.search(reg, datapoint)[1]
    except TypeError:
        found_section = ""
    return found_section


def find_list_of_cells(datapoint):
    reg = r"<cell> (.*?) </cell>"
    return re.findall(reg, datapoint)


def find_cell_value(cell):
    # print(cell)
    reg = r"(.*) <col_header>"
    try:
        found_value = re.search(reg, cell)[1]
    except TypeError:
        found_value = cell
    return found_value


def find_column_value(cell):
    reg = r"<col_header> (.*) </col_header>"
    try:
        found_col = re.search(reg, cell)[1]
    except TypeError:
        found_col = ""
    return found_col


def extract_type(cell_value):
    if is_numeric(cell_value):
        return "NUMERIC"
    elif is_date(cell_value):
        return "DATE"
    else:
        return "STRING"


def extract_totto_datapoint_attributes(datapoint):
    cells = find_list_of_cells(datapoint)
    cells_info = []
    for cell in cells:
        cell_value = find_cell_value(cell)
        cell_col = find_column_value(cell)
        cell_type = extract_type(cell_value)

        cells_info.append({"value": cell_value, "col": cell_col, "type": cell_type})

    return {
        "title": find_table(datapoint),
        "query": find_section(datapoint),
        "cells": cells_info,
    }


def datapoint_attributes_to_compact(datapoint):
    datapoint_attributes = extract_totto_datapoint_attributes(datapoint)

    table_content = ""
    for ind, cell in enumerate(datapoint_attributes["cells"]):

        def serialize_cell():
            return f"<col{ind}> {cell['col']} | {cell['type']} | {cell['value']} "

        table_content += serialize_cell()

    return f"<query> {datapoint_attributes['query']} <table> {datapoint_attributes['title']} {table_content}"


def to_compact(datapoints):
    return [
        {
            "subtable_and_metadata": datapoint_attributes_to_compact(
                datapoint["subtable_and_metadata"]
            ),
            "final_sentence": datapoint["final_sentence"],
        }
        for datapoint in datapoints
    ]
