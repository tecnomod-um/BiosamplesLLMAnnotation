import json #use json data
import pandas as pd #dataframe manipulation

from pattern_analysis import df_to_dicc 

count=0 #get the count of related identifiers

def data_process(filename):
    """
    Load data from a JSON file and convert it to a structured DataFrame.

    Parameters:
        filename (str): Path to the JSON file containing ontology class data.

    Returns:
        pd.DataFrame: A processed DataFrame with columns for different ontology classes and labels.
    """

    with open(filename, 'r') as archivo:
        dict_classes = json.load(archivo)
    df = pd.DataFrame.from_dict(dict_classes, orient='index')
    new_row = pd.DataFrame([df.columns], columns=df.columns)
    df_process = pd.concat([new_row, df], ignore_index=True)
    df_process.columns = ['CLO_C','CLO_M', 'CL_C','CL_M', 'UBERON_C','UBERON_M','BTO_C','BTO_M']
    df_process = df_process.drop(0)

    keys=[]
    for key in dict_classes.keys():
        keys.append(key)
    df_process['Label'] = keys #add label column
    
    df_process.fillna("", inplace=True) #replace 'Nonetype' values

    return df_process


def contribution(type):
    """
    Process potential contributions by a language model (LLM) to fill missing ontology data
    for each identifier.

    Parameters:
        type (str): The ontology type ('CL', 'CT', or 'A') to specify which data to process.

    """
    global count 

    if type == 'CL':
        df = data_process('pattern_file_CL.json')
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    elif type == 'CT':
        df = data_process('pattern_file_CT.json')
        suffixes = ['CL', 'UBERON', 'BTO']
    elif type == 'A':
        df = data_process('pattern_file_A.json')
        suffixes = ['UBERON', 'BTO']
    else:
        raise ValueError("Type not recognized")

    for suffix in suffixes:
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        for index,row in df.iterrows():
            ref=row[true_col]
            pred = row[pred_col]
            if ref == '-' and pred != '-':
                print(index,suffix,row) 
                check = input('Is the contribution valid? Y/N \n')
                if check == 'Y':
                    df.at[index, true_col] = pred
                    count=count+1
    dicc = df_to_dicc(df)
    file_name = f'contribution_file_{type}.json'
    with open(file_name, 'w') as archive_json:
        json.dump(dicc, archive_json, indent=4)
    return 

def main():
    contribution('A')
    contribution('CT')
    contribution('CL')
    print("Number of valid contributions:", count)

if __name__ == "__main__":
    main()


