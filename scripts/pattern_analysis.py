import json #use json data
import pandas as pd #dataframe manipulation 

count=0 #get the count of related identifiers

def data_process(filename):
    """
    Load data from a JSON file and convert it to a structured DataFrame.

    Parameters:
        filename (str): Path to the JSON file containing class data.

    """
    with open(filename, 'r') as archive:
        dict_classes = json.load(archive)
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

def search_common_substrings(string1, string2):
    """
    Find the longest common substring between two strings.

    Parameters:
        string1 (str): First string to compare.
        string2 (str): Second string to compare.

    """
    length = min(len(string1), len(string2))
    for i in range(length, 0, -1):
        for j in range(len(string1) - i + 1):
            if string1[j:j+i] in string2:
                return string1[j:j+i]
    return ""

def check_pattern(df, index, column, pattern, string1, string2):
    """
    Check if a common pattern (substring) between two strings meets criteria and update the DataFrame if valid.

    Parameters:
        df (pd.DataFrame): The DataFrame to update.
        index (int): Row index in the DataFrame.
        column (str): Column name to update if the pattern is valid.
        pattern (str): The common pattern found between the two strings.
        string1 (str): First string for comparison.
        string2 (str): Second string for comparison.

    """
    global count 
    print(column)
    if len(pattern) > 4 and pattern != ' cell' and pattern != ' of ':
        print(index,'The similarity between', string1,'-', string2,':', pattern, len(pattern))
        check = input('Is the pattern valid? Y/N \n')
        if check == 'Y':
            df.at[index, column] = string1
            count +=1
    return df

def df_to_dicc(df):
    """
    Convert a DataFrame to a dictionary with labels as keys and identifiers as values.

    Parameters:
        df (pd.DataFrame): Input DataFrame to convert.

    """
    dicc={}
    for index,row in df.iterrows():
        label = row['Label']
        identifiers = row[:8].tolist()
        dicc[label]=identifiers
    return dicc

def pattern_process(type):
    """
    For columns that share a pattern, it is checked if the pattern is valid and save the results to JSON.

    Parameters:
        type (str): The ontology type ('CL', 'CT', or 'A') to specify which data to process.

    """
    # Load specific file and set columns of interest based on type
    if type == 'CL':
        df = data_process('classnames_CL.json')
        col_1 = 'CLO'
        col_2 = 'BTO'
    elif type == 'CT':
        df = data_process('classnames_CT.json')
        col_1 = 'CL'
        col_2 = 'BTO'
    elif type == 'A':
        df = data_process('classnames_A.json')
        col_1 = 'UBERON'
        col_2 = 'BTO'
    else:
        raise ValueError("Tipo no reconocido")

    # Define control and target columns based on type
    control_1= f'{col_1}_C'
    control_2= f'{col_2}_C'
    test_1= f'{col_1}_M'
    test_2= f'{col_2}_M'

    for index, row in df.iterrows():
        string1a = row[control_1]
        string2a = row[test_1]
        if string1a != string2a:
            pattern = search_common_substrings(string1a, string2a)
            pattern_df = check_pattern(df, index, test_1, pattern, string1a, string2a)
    
    for index, row in pattern_df.iterrows():
        string1b = row[control_2]
        string2b = row[test_2]
        if string1b != string2b:
            pattern = search_common_substrings(string1b, string2b)
            pattern2_df =check_pattern(pattern_df, index, test_2, pattern, string1b, string2b)
    dicc = df_to_dicc(pattern2_df)
    file_name = f'pattern_file_{type}.json'
    with open(file_name, 'w') as archive_json:
        json.dump(dicc, archive_json, indent=4)

def main():
    pattern_process('A')
    pattern_process('CL')
    pattern_process('CT')
    print("The pattern was valid", count, "times.")

if __name__ == "__main__":
    main()


