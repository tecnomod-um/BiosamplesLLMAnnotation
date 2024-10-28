import json #use json data
import pandas as pd #dataframe manipulation 

from creation_ft import mappings_test #get test data from previous script

def process_json_results(file_path):
    """
    Convert the archive result.json obtained from the different models into a DataFrame. It also quantized the cases where the model
    do not use the correct output format.

    Parameters:
        file_path (str): Path to the results to the model.

    """

    with open(file_path, 'r') as archivo:
        results_model = json.load(archivo)

    l_l = []
    for label, identifiers in results_model.items():
        list_elements = identifiers.strip('][').split(', ')  # Convert to list
        list_elements = [element.strip("'") for element in list_elements]
        list_elements.insert(0, label)
        l_l.append(list_elements)
    
    l_l_filtered = [sublist for sublist in l_l if len(sublist) == 5] #get the outputs in the correct format
    mappings_model = pd.DataFrame(l_l_filtered,columns=["Label", "CLO_M", "CL_M", "UBERON_M", "BTO_M"])
    model_error = [sublist for sublist in l_l if len(sublist) != 5] #get the outputs in the wrong format
    
    print('Total of correct data outputs:', len(mappings_model))
    print('The following number of model outputs do not meet the required format:', len(model_error))
    return mappings_model

def get_df_comparation(path,mappings_test):
    """
    Obtain a unique DataFrame for comparing reference identifiers and the model identifiers.

    Parameters:
        path (str): Path for the result of a model.
        mappings_test (DataFrame): DataFrame containing the labels to be mapped with the reference mappings.

    """
    mappings_model = process_json_results(path) #obtain model mappings dataframe
    mappings_control = mappings_test #obtain reference mappings dataframe
    mappings_control.columns = ['Label', 'CLO_C', 'CL_C', 'UBERON_C', 'BTO_C', 'Type']

    df_comparation = pd.merge(mappings_model, mappings_control, on='Label', how='inner') #merge both dataframes
    sorted_columns= ["Label","Type","CLO_C", "CLO_M","CL_C","CL_M","UBERON_C", "UBERON_M","BTO_C","BTO_M"]
    df_comparation = df_comparation[sorted_columns]

    for column in df_comparation:
        df_comparation[column] = df_comparation[column].fillna('unknown') #replace na values with the string 'unknown'
    return df_comparation

df_comparation_gpt3_5= get_df_comparation('results__gpt3_5.json',mappings_test)
df_comparation_gpt4= get_df_comparation('results__gpt4.json',mappings_test)
df_comparation_gpt4o= get_df_comparation('results__gpt4_o.json',mappings_test)
df_comparation_ft= get_df_comparation('results_ft_35.json',mappings_test)
df_comparation_ft_4o= get_df_comparation('results_ft_4o.json',mappings_test)




