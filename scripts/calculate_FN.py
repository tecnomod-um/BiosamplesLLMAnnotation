import json  # use json data
import pandas as pd  # dataframe

from class_names import df_A,df_CL,df_CT

count_false_negatives = 0  # get the count of related identifiers

def calculate_fn(df,type):
    """
    Process potential contributions by a language model (LLM) to fill missing ontology data
    for each identifier.

    Parameters:
        type (str): The ontology type ('CL', 'CT', or 'A') to specify which data to process.

    """
    global count_false_negatives

    if type == 'CL':
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    elif type == 'CT':
        suffixes = ['CL', 'UBERON', 'BTO']
    elif type == 'A':
        suffixes = ['UBERON', 'BTO']
    else:
        raise ValueError("Type not recognized")

    for suffix in suffixes:
        count_false_negatives = 0
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        for index, row in df.iterrows():
            ref = row[true_col]
            pred = row[pred_col]
            if ref != '-' and pred == '-':
                count_false_negatives = count_false_negatives + 1
        print(type,"For the ontology ",suffix,"the number of FN is: ",count_false_negatives)
    return

def main():
    calculate_fn(df_A,"A")
    calculate_fn(df_CL, "CL")
    calculate_fn(df_CT, "CT")

if __name__ == "__main__":
    main()


