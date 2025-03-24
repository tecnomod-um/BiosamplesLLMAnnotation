import requests #send HTTP requests
import json #use json data
from dotenv import dotenv_values
import pandas as pd

def load_data(data_path):
    """
    Load the CSV file into a DataFrame.
    :param data_path: Path to the CSV file.
    :return: A pandas DataFrame containing the data.
    """
    return pd.read_csv(data_path, sep=",", header=0)

def load_environment():
    """
    Load environment variables.
    :return: The OPENAI API key.
    """
    config = dotenv_values(dotenv_path=".env")
    return config.get('BIOPORTAL_API_KEY')

def group_and_return_dfs(df):
    """
    Group the dataframe by the 'Type' column and create a dictionary with a dataframe for each group.

    Parameters:
        df (DataFrame): The input dataframe with a 'Type' column for grouping.

    Returns:
        dict: Dictionary with keys as group names and values as dataframes for each group.
    """
    dfs = {}
    for group_name, group_df in df.groupby('Type'):
        dfs[group_name] = group_df
    return dfs

def get_class_name(ontology_acronym,class_id):
    """
     Retrieve the class name from BioPortal API for a given class identifier.

     Parameters:
         ontology_acronym (str): Acronym for the ontology (e.g., 'CL', 'CLO').
         class_id (str): The identifier for the class within the ontology.

     Returns:
         str: The name of the class if found, or None if the request fails.
     """
    url = f'http://data.bioontology.org/ontologies/{ontology_acronym}/classes/{class_id}'
    api_key = load_environment()
    headers = {
        'Authorization': f'apikey token={api_key}',
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        class_name = data.get('prefLabel')  # or another key based on the structure
        print("Class Name:", class_name)
        return class_name
    else:
        print("Failed to retrieve data:", response.status_code, class_id)

def get_classes(df):
    """
    Retrieve class names for each identifier in the dataframe, based on ontology acronyms.

    Parameters:
        df (dict): Dictionary where keys are labels, and values are lists of identifiers.

    Returns:
        dict: Dictionary where keys are labels, and values are lists of class names.
    """
    dicc_clases={}
    for label,identifiers in df.items():      
        for element in identifiers:
            element_formatted=element.replace("_",":") #change underscore for a colon
            if element_formatted.startswith("CL:"):
                class_name=get_class_name("CL",element_formatted)
            elif element_formatted.startswith("CLO:"):
                class_name=get_class_name("CLO",element_formatted)
            elif element_formatted.startswith("UBERON:"):
                class_name=get_class_name("UBERON",element_formatted)
            elif element_formatted.startswith("BTO:"):
                class_name=get_class_name("BTO",element_formatted)
            else:
                class_name="-"
            if label in dicc_clases:
                dicc_clases[label].append(class_name)
            else:
                dicc_clases[label]=[class_name]
    return dicc_clases

def df_to_dicc(df):
    """
    Convert a DataFrame to a dictionary where labels map to lists of identifiers.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        dict: Dictionary with labels as keys and lists of identifiers as values.
    """
    dicc={}
    for index,row in df.iterrows():
        label = row['Label']
        identifiers = row[2:].tolist()
        dicc[label]=identifiers
    return dicc

def get_class_names(df,type):
    """
    Get class names from identifiers in the dataframe and save them in JSON format.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        type (str): A string used for the JSON file name to specify the type of data.

    Returns:
        None: Writes output to a JSON file.
    """
    dicc_df = df_to_dicc(df)
    print(dicc_df)
    dicc_clases = get_classes(dicc_df)
    file_name = f'classnames_{type}.json'
    with open(file_name, 'w') as json_file:
        json.dump(dicc_clases, json_file, indent=4)

df_comparison_ft_4o_mini = load_data("../results/df_ft_4o_mini_annotation.csv")
dfs = group_and_return_dfs(df_comparison_ft_4o_mini)

df_CL = dfs['CL']  # DataFrame where Type is 'CL'
df_CT = dfs['CT']  # DataFrame where Type is 'CT'
df_A = dfs['A']    # DataFrame where Type is 'A'
df_dash = dfs['-']    # DataFrame where Type is '-'

def main():
     get_class_names(df_A,'A')
     get_class_names(df_CL,'CL')
     get_class_names(df_CT,'CT')

if __name__ == "__main__":
    main()

