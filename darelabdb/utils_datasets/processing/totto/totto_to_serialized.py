from darelabdb.utils_datasets.processing.totto import preprocessing_utils


def generate_serialized_table(
    datapoint, full_table: bool = False, with_metadata: bool = True
) -> str:
    table = datapoint["table"]
    table_page_title = datapoint["table_page_title"]
    table_section_title = datapoint["table_section_title"]
    cell_indices = datapoint["highlighted_cells"]

    if full_table and not with_metadata:
        return preprocessing_utils.linearize_full_table(
            table=table,
            cell_indices=cell_indices,
            table_page_title=None,
            table_section_title=None,
        )
    elif full_table and with_metadata:
        return preprocessing_utils.linearize_full_table(
            table=table,
            cell_indices=cell_indices,
            table_page_title=table_page_title,
            table_section_title=table_section_title,
        )
    elif not full_table and not with_metadata:
        subtable = preprocessing_utils.get_highlighted_subtable(
            table=table, cell_indices=cell_indices, with_heuristic_headers=True
        )

        return preprocessing_utils.linearize_subtable(
            subtable=subtable, table_page_title=None, table_section_title=None
        )
    elif not full_table and with_metadata:
        subtable = preprocessing_utils.get_highlighted_subtable(
            table=table, cell_indices=cell_indices, with_heuristic_headers=True
        )

        return preprocessing_utils.linearize_subtable(
            subtable=subtable,
            table_page_title=table_page_title,
            table_section_title=table_section_title,
        )
