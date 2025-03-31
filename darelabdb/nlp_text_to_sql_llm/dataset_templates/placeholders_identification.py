import pandas as pd
import re


def _get_placeholder_names_in_string(text: str) -> list[str]:
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, text)
    return matches


def placeholders_identification(templates: list[str]) -> pd.DataFrame:
    """
    Prints the information about the placeholders existing in the given templates.

    Args:
        templates: A list of texts that contain placeholder values (e.g., "Lorem ipsum dolor {placeholder_1} amet")

    Returns:
         A dataframe with:

            - placeholders: All the placeholders existing in the templates.
            - dependencies: All the coexisting placeholders for each placeholder in the given list of templates.
    """
    all_placeholders = set()
    dependencies = {}

    for template in templates:
        placeholders = _get_placeholder_names_in_string(template)
        # Update the list with all the existing placeholders
        all_placeholders.update(placeholders)

        # Update placeholders' dependencies
        if len(placeholders) > 1:
            for placeholder in placeholders:
                coexisting_placeholders = set(placeholders)
                coexisting_placeholders.remove(placeholder)
                if placeholder not in dependencies:
                    dependencies[placeholder] = coexisting_placeholders
                else:
                    dependencies[placeholder].update(coexisting_placeholders)

    # Add placeholders with zero dependencies in the dependencies dictionary
    for placeholder in all_placeholders:
        if placeholder not in dependencies:
            dependencies[placeholder] = set()

    return pd.DataFrame(
        {"name": list(dependencies.keys()), "dependencies": list(dependencies.values())}
    )


if __name__ == "__main__":
    query_templates = pd.read_excel(
        "development/faircore_nl_search/storage/faircore_final_benchmark.xlsx",
        sheet_name="Benchmark",
    )[["sql_query_without_values", "nl_question_without_values"]]

    placeholders = placeholders_identification(
        query_templates["sql_query_without_values"].dropna().tolist()
    )

    placeholders_nl = placeholders_identification(
        query_templates["nl_question_without_values"].dropna().tolist()
    )
