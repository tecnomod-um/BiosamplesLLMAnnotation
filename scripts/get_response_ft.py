from openai import OpenAI #ChatGPT API
from dotenv import dotenv_values #environment control
import json #use json data
import time

from creation_ft import mappings_test #get test data from previous script

def load_environment(var):
    """
    Get the environment variables, in this case, the OPENAI API key and the ID of the fine-tuned models.

    Parameters:
        var (str): Variable to be exported.
    """
    config = dotenv_values(dotenv_path="../.env")
    if var == 'key':
        return config['OPENAI_API_KEY']
    if var == 'ft_model_4o':
        return config['FT_MODEL_4o']
    if var == 'ft_model_4o_mini':
        return config['FT_MODEL_4o_mini']


def get_openai_response(df,model):
    """
    Get the output from the fine-tuned model.

    Parameters:
        df (DataFrame): DataFrame containing the labels to be mapped.
        model (str): Fine-tuned model to which the consultation is to be made.
    """
    df.columns = ['Label', 'CLO', 'CL', 'UBERON', 'BTO', 'Type']
    api_key = load_environment('key')
    client = OpenAI(api_key=api_key)
    dicc = {}
    for index, row in df.iterrows():
        label = row['Label']
        completion = client.chat.completions.create(
            model=model, #fine-tuning model
            messages=[
                {"role": "system", "content": "You are going to assist me in a search of the identifiers of ontologies for a determined label."},
                {"role": "user", "content": f"For the label {label}, I need you to search the identifiers that better suit the label in the ontologies CLO, CL, UBERON, and BTO."}
            ]
        )
        out = completion.choices[0].message.content
    
        dicc[label] = out
    return dicc

def save_results(results,name):
    """
    Save the output of the model in JSON format.

    Parameters:
        results (dict): Dictionary containing the label and its correspondings identifiers for each of the ontologies of interest.
        name (str): Name for the new JSON file.
    """
    with open(name, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def main():
    results_4o = get_openai_response(mappings_test,model=load_environment('ft_model_4o_mini'))
    save_results(results_4o,'results_ft_4o_mini.json')

if __name__ == "__main__":
    start_time = time.time()  # Start the timer
    main()  # Execute the main function
    end_time = time.time()  # Stop the timer
    print(f"Execution time: {end_time - start_time} seconds")