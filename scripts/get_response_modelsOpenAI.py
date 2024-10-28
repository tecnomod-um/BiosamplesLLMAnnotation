from openai import OpenAI #ChatGPT API
import json #use json data
from dotenv import dotenv_values #environment control

from creation_ft import mappings_test #get test data from previous script

def load_environment():
    """
    Get the environment variables, in this case, the OPENAI API key.
    """
    config = dotenv_values(dotenv_path=".env")
    return config['OPENAI_API_KEY']

def read_prompt_file(file_path):
    """
    Read the prompt file.
    Parameters:
        file_path (str): path to the txt file that contains the instructions for the model.
    """
    with open(file_path, 'r') as file:
        return file.read()

def format_prompt(prompt, label):
    """
    Format the prompt file with the desired label.

    Parameters:
        prompt (str): Prompt with detailed instructions for the task to be performed by the OpenAI model.
        label (str): Label to be mapped to the ontologies of interest.
    """
    return prompt.format(label=label)

def get_openai_response(df,model):
    """
    Get the output from the OpenAI base model.

    Parameters:
        df (DataFrame): DataFrame containing the label to be mapped.
        model (str): OpenAI base model to which the consultation is to be made.
    """
    df.columns = ['Label', 'CLO', 'CL', 'UBERON', 'BTO', 'Type']
    api_key=load_environment()
    client = OpenAI(api_key=api_key)
    dicc = {}
    for index,row in df.iterrows():
        label= row[0]
        prompt = read_prompt_file('prompt_search_id.txt')
        f_prompt = format_prompt(prompt,label)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are going to assist me in a search of the identifiers of ontologies for a determined label."},
                {"role": "user", "content": f_prompt }
            ]
        )
        out=completion.choices[0].message.content
        if label in dicc.keys():
            dicc[label].append(out)
        else:
            dicc[label]= out
    return dicc

def save_results(results,name):
    """
    Save the output of the model in JSON format.

    Parameters:
        results (dict): Dictionary containing the label and its correspondings identifiers for each of the ontologies of interest.
        name (str): Name for the new JSON file.
    """
    with open(name, 'w') as archivo_json:
        json.dump(results, archivo_json, indent=4)

def main():
    results_3_5 = get_openai_response(mappings_test,"gpt-3.5-turbo-0125")
    results_4 = get_openai_response(mappings_test,"gpt-4-turbo")
    results_4o = get_openai_response(mappings_test,"gpt-4o")
    save_results(results_3_5,'results__gpt3_5.json')
    save_results(results_4,'results__gpt4.json')
    save_results(results_4o,'results__gpt4_o.json')

if __name__ == "__main__":
    main()

