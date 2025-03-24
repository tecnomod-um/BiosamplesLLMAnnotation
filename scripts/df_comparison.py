import json #use json data
import pandas as pd #dataframe manipulation
from pandas import read_csv

def process_json_results(file_path):
    """
    Convert the archive results.json obtained from the different models into a DataFrame. It also quantized the cases where the model
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

def get_df_comparison(path, mappings_test):
    """
    Obtain a unique DataFrame for comparing reference identifiers and the model identifiers.

    Parameters:
        path (str): Path for the results of a model.
        mappings_test (DataFrame): DataFrame containing the labels to be mapped with the reference mappings.

    """
    mappings_model = process_json_results(path) #obtain model mappings dataframe
    mappings_control = mappings_test #obtain reference mappings dataframe
    mappings_control.columns = ['Label', 'CLO_C', 'CL_C', 'UBERON_C', 'BTO_C', 'Type']

    df_comparison = pd.merge(mappings_model, mappings_control, on='Label', how='inner') #merge both dataframes
    sorted_columns= ["Label","Type","CLO_C", "CLO_M","CL_C","CL_M","UBERON_C", "UBERON_M","BTO_C","BTO_M"]
    df_comparison = df_comparison[sorted_columns]

    for column in df_comparison:
        df_comparison[column] = df_comparison[column].fillna('unknown') #replace na values with the string 'unknown'
    return df_comparison

mappings_test = read_csv('./finetuning_process/mappings_test.csv',header=0)
df_comparison_gpt35= get_df_comparison('results/results__gpt3_5.json', mappings_test)
df_comparison_gpt4= get_df_comparison('results/results__gpt4.json', mappings_test)
df_comparison_gpt4o= get_df_comparison('results/results__gpt4_o.json', mappings_test)
df_comparison_ft_gpt35= get_df_comparison('results/results_ft_35.json', mappings_test)
df_comparison_ft_gpt4o= get_df_comparison('results/results_ft_4o.json', mappings_test)
df_comparison_ft_gpt4o_mini= get_df_comparison('results/results_ft_4o_mini.json', mappings_test)
df_comparison_ft_gpt4o_mini.to_csv('./results/df_ft_4o_mini_annotation.csv',index=False)
df_comparison_ft_gpt4o_mini_descriptions= read_csv("./results/df_ft_4o_mini_descriptions.csv",header=0)



