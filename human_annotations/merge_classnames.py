import pandas as pd
import json
from typing import List

# Fixed column order matching the JSON value lists
ONTOLOGY_COLUMNS: List[str] = ['CLO_C', 'CLO_M', 'CL_C', 'CL_M', 'UBERON_C', 'UBERON_M', 'BTO_C', 'BTO_M']


def load_json_to_dataframe(filepath: str, type_label: str) -> pd.DataFrame:
    """
    Load a JSON file and convert it into a pandas DataFrame, mapping each list of values
    to predefined ontology columns.

    Args:
        filepath (str): Path to the JSON file.
        type_label (str): A label to identify the source/type of each entry.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Label', 'Type'] + ontology columns.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    rows = []
    for label, values in data.items():
        if isinstance(values, list) and len(values) == len(ONTOLOGY_COLUMNS):
            row = {'Label': label, 'Type': type_label}
            row.update(dict(zip(ONTOLOGY_COLUMNS, values)))
            rows.append(row)
        else:
            print(f"⚠️ Label '{label}' does not contain exactly {len(ONTOLOGY_COLUMNS)} elements.")

    return pd.DataFrame(rows)


def main():
    # Define input file paths and their corresponding type labels
    file_mappings = [
        ('results/classnames_A_e2.json', 'A'),
        ('results/classnames_CL_e2.json', 'CL'),
        ('results/classnames_CT_e2.json', 'CT')
    ]

    # Load and combine all datasets
    dataframes = [load_json_to_dataframe(path, label) for path, label in file_mappings]
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Reorder columns for consistency
    combined_df = combined_df[['Label', 'Type'] + ONTOLOGY_COLUMNS]

    # Optionally, save the final DataFrame to a CSV file
    combined_df.to_csv('classnames_e2.csv', index=False)


if __name__ == "__main__":
    main()
