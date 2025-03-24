import pandas as pd #dataframe manipulation
from sklearn.model_selection import train_test_split #data division 
from openai import OpenAI #ChatGPT API
from dotenv import dotenv_values #environment control
import os #interact with the operating system
import json #use json data

mappings = pd.read_csv("biosamples.tsv", sep="\t", header=None) #data loading
mappings_ft, mappings_test = train_test_split(mappings, test_size=0.30, random_state=17) #first data division
mappings_train, mappings_validation = train_test_split(mappings_ft, test_size=0.25, random_state=17) #second data
mappings_test.to_csv('mappings_test.csv',index=False)

def get_formatted_data(data):
    """
    Get the correct format for the fine-tuning.

    Parameters:
        data (Dataframe): DataFrame to be transformed into the message format (JSON) required by OpenAI for fine-tuning.
    """
    formatted_data = [] #list to store the messages
    for row in data.itertuples(index=False):
        label = row[0]
        identifiers = list(row[1:5])
        formatted_data.append({
            "messages": [
                {"role": "system", "content": "You are going to assist me in a search of the identifiers of ontologies for a determined label."},
                {"role": "user", "content": f"For the label {label}, I need you to search the identifiers that better suit the label in the ontologies CLO, CL, UBERON and BTO."},
                {"role": "assistant", "content": str(identifiers)}
            ]
        })
    return formatted_data

def save_to_jsonl(dataset, file_path):
    """
    Convert a list of messages in JSON format to JSONL.

    Parameters:
        file_path (str): Path to the folder where the training and validation data will be stored.
        dataset (list): List of message to be converted to JSON format.

    """
    with open(file_path, 'w') as file:
        for example in dataset:
            json_line = json.dumps(example)
            file.write(json_line + '\n')

def jsonl_converter(mappings_train, mappings_validation, output_folder):
    """
    Convert a list of messages to JSONL.

    Parameters:
        mappings_train (DataFrame): DataFrame to be used as training dataset.
        mappings_validation (DataFrame): DataFrame to be used as validation dataset.
        output_folder (str): Path to the folder where the training and validation data will be stored.
    """
    formatted_train = get_formatted_data(mappings_train)
    formatted_validation = get_formatted_data(mappings_validation)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define file paths
    training_file_path = os.path.join(output_folder, "formatted_train.jsonl")
    validation_file_path = os.path.join(output_folder, "formatted_validation.jsonl")

    # Save the files
    save_to_jsonl(formatted_train, training_file_path)
    save_to_jsonl(formatted_validation, validation_file_path)

def load_environment():
    """
        Get the environment variables, in this case, the OPENAI API key.
    """
    config = dotenv_values(dotenv_path="../.env")
    return config['OPENAI_API_KEY']

def prepare_data_ft():
    """
    Upload the training and validation files to the OpenAI platform, indicating that the purpose is performing a fine-tuning job.
    """
    api_key=load_environment()
    client = OpenAI(api_key=api_key)

    training_file_id = client.files.create(
    file=open("../formatted_train.jsonl", "rb"),
    purpose="fine-tune")

    validation_file_id = client.files.create(
    file=open("../formatted_validation.jsonl", "rb"),
    purpose="fine-tune")

    print(f"Training File ID: {training_file_id}")
    print(f"Validation File ID: {validation_file_id}")
    return client,training_file_id,validation_file_id

def create_job():
  """
  Create the fine-tuning job with the selected model and the provided training and validation data.
  """
  client,training_file_id,validation_file_id = prepare_data_ft()
  response = client.fine_tuning.jobs.create(
    training_file=training_file_id.id,
    validation_file =validation_file_id.id,
    model="gpt-4o-2024-08-06", #"gpt-4o-mini-2024-07-18"
    suffix='4o_ft_annotation',
    hyperparameters={
    "n_epochs": 6,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
    }
  )
  job_id = response.id
  status = response.status
  print(f'Fine-tuned model with jobID: {job_id}.')
  print(f"Training Response: {response}")
  print(f"Training Status: {status}")

  return

def main(mappings_train,mappings_validation,output_folder):
    jsonl_converter(mappings_train, mappings_validation, output_folder)
    #create_job()

if __name__ == "__main__":
    output_folder = input('Path to the folder where the training and validation data will be stored:')
    main(mappings_train,mappings_validation,output_folder)
