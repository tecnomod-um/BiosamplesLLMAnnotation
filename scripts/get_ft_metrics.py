from dotenv import dotenv_values #environment control
from openai import OpenAI #ChatGPT API
import base64

def load_environment(var):
    """
    Get the environment variables, in this case, the OPENAI API key and the ID of the fine-tuned models.

    Parameters:
        var (str): Variable to be exported.
    """
    config = dotenv_values(dotenv_path=".env")
    if var == 'key':
        return config['OPENAI_API_KEY']
    if var == 'ft_model_4o_id':
        return config['FT_MODEL_4o_ID']
    if var == 'ft_model_35_id':
        return config['FT_MODEL_35_ID']

def get_metrics(client,ft_model_id,name):
    """
    Get a CSV file with different metrics to evaluate the training of the model during the fine-tuning process.

    Parameters:
        ft_model_id (str): ID of the fine-tuned model to get metrics.
        name (str): Name for CSV file with the metrics of the fine-tuned model.
    """
    fine_tune_results = client.fine_tuning.jobs.retrieve(ft_model_id).result_files
    result_file = client.files.retrieve(fine_tune_results[0])
    content = client.files.content(result_file.id)
    with open(name, "wb") as f:
        f.write(base64.b64decode(content.text.encode('utf-8')))

def main():
    api_key = load_environment('key')
    client = OpenAI(api_key=api_key)
    ft_4o_model_id = load_environment('ft_model_4o_id')
    ft_35_model_id = load_environment('ft_model_35_id')
    get_metrics(client,ft_4o_model_id, "result_4o.csv")
    get_metrics(client, ft_35_model_id, "result_35.csv")

if __name__ == "__main__":
    main()