import json  # use json data
import pandas as pd  # dataframe manipulation
import matplotlib.pyplot as plt  # data visualization

from class_names import df_dash

def data_process(filename):
    """
    Load data from a JSON file and convert it to a structured DataFrame.

    Parameters:
        filename (str): Path to the JSON file containing ontology class data.

    """
    with open(filename, 'r') as archivo:
        dict_classes = json.load(archivo)
    df = pd.DataFrame.from_dict(dict_classes, orient='index')
    new_row = pd.DataFrame([df.columns], columns=df.columns)
    df_process = pd.concat([new_row, df], ignore_index=True)
    df_process.columns = ['CLO_C', 'CLO_M', 'CL_C', 'CL_M', 'UBERON_C', 'UBERON_M', 'BTO_C', 'BTO_M']
    df_process = df_process.drop(0)

    keys = []
    for key in dict_classes.keys():
        keys.append(key)
    df_process['Label'] = keys  # add label column

    df_process.fillna("", inplace=True)  # replace 'Nonetype' values

    return df_process


def match_calculation(type):
    """
    Calculate the perfect match ratio for each ontology by comparing columns for expected and predicted values.

    Parameters:
        type (str): The ontology type ('CL', 'CT', 'A', or 'dash').

    """
    if type == 'CL':
        df = data_process('contribution_file_CL.json')
        col_1 = 'CLO'
        col_2 = 'BTO'
    elif type == 'CT':
        df = data_process('contribution_file_CT.json')
        col_1 = 'CL'
        col_2 = 'BTO'
    elif type == 'A':
        df = data_process('contribution_file_A.json')
        col_1 = 'UBERON'
        col_2 = 'BTO'
    elif type == 'dash':
        return 0  # Set perfect match to 0 for 'dash'
    else:
        raise ValueError("Tipo no reconocido")

    perfect_match = 0
    no_perfect_match = 0

    control_1 = f'{col_1}_C'
    control_2 = f'{col_2}_C'
    test_1 = f'{col_1}_M'
    test_2 = f'{col_2}_M'

    for index, row in df.iterrows():
        string1a = row[control_1]
        string2a = row[test_1]
        string1b = row[control_2]
        string2b = row[test_2]
        if string1a == string2a and string1b == string2b:
            perfect_match += 1
        else:
            no_perfect_match += 1

    total = perfect_match + no_perfect_match
    index_pm = perfect_match / total
    return index_pm


def calculate_metrics(type):
    """
    Calculate precision, exhaustiveness, and F1-score for each ontology type.

    Parameters:
        type (str): The ontology type ('CL', 'CT', 'A', or 'dash').

    """
    if type == 'CL':
        df = data_process('contribution_file_CL.json')
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
        false_neg_data = {'CLO': 110, 'CL': 52, 'UBERON': 26, 'BTO': 121}
    elif type == 'CT':
        df = data_process('contribution_file_CT.json')
        suffixes = ['CL', 'UBERON', 'BTO']
        false_neg_data = {'CLO': 0, 'CL': 15, 'UBERON': 11, 'BTO': 63}
    elif type == 'A':
        df = data_process('contribution_file_A.json')
        suffixes = ['UBERON', 'BTO']
        false_neg_data = {'CLO': 0, 'CL': 0, 'UBERON': 1, 'BTO': 18}
    elif type == 'dash':
        df = df_dash
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
        false_neg_data = None
    else:
        raise ValueError("Tipo no reconocido")

    accuracies = {}
    exhaust = None if type == 'dash' else {}
    f1 = None if type == 'dash' else {}

    for suffix in suffixes:
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        true_pos = 0
        false_pos = 0
        false_neg = 0 if false_neg_data is None else false_neg_data[suffix]
        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                if df.iloc[i][true_col] == df.iloc[i][pred_col]:
                    true_pos += 1
                else:
                    false_pos += 1
            true_pos = true_pos - false_neg
            if type != 'dash':
                exhaust_values = true_pos / (true_pos + false_neg)
                f1_values = 2 * true_pos / (2 * true_pos + false_neg + false_pos)
                exhaust[suffix] = exhaust_values
                f1[suffix] = f1_values
            accuracy = true_pos / (true_pos + false_pos)
            accuracies[suffix] = accuracy
            print(type,suffix,false_pos,true_pos)
        else:
            if type != 'dash':
                exhaust[suffix] = None
                f1[suffix] = None
            accuracies[suffix] = None
    return accuracies, exhaust, f1


def plot_combined_metrics():
    """
    Plot the calculated metrics for precision, exhaustiveness, and F1-score for all ontology types (CL, CT, A).
    """
    types = ['CL', 'CT', 'A', 'dash']
    metrics_data = {'precision': {}, 'exhaustiveness': {}, 'f1_score': {}, 'perfect_match': {}}

    for type in types:
        precision, exhaust, f1 = calculate_metrics(type)
        metrics_data['precision'][type] = precision
        metrics_data['exhaustiveness'][type] = exhaust
        metrics_data['f1_score'][type] = f1
        metrics_data['perfect_match'][type] = match_calculation(type)

    # Plot precision
    df_precision = pd.DataFrame(metrics_data['precision']).T
    df_precision['Perfect_match'] = metrics_data['perfect_match']
    ax = df_precision.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_title('Precision by Type')
    ax.set_xlabel('Types')
    ax.set_ylabel('Precision')
    plt.legend(title='Ontologies')
    plt.xticks(ticks=range(len(df_precision.index)), labels=df_precision.index, ha='center', rotation=0, fontsize=10)

    # Add value labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

    plt.subplots_adjust(bottom=0.4)  # Increase space between bars to avoid overlap
    plt.show()

    # Plot exhaustiveness
    types_exhaust_f1 = ['CL', 'CT', 'A']  # Remove 'dash' from exhaustiveness and F1-score plots
    df_exhaust = pd.DataFrame({key: metrics_data['exhaustiveness'][key] for key in types_exhaust_f1}).T
    ax = df_exhaust.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    ax.set_ylim(0, 1.2)
    ax.set_title('Recall by Type')
    ax.set_xlabel('Types')
    ax.set_ylabel('Recall')
    plt.legend(title='Ontologies', loc='upper left',
               bbox_to_anchor=(1, 1))  # Move legend outside the plot to avoid overlap
    plt.xticks(ticks=range(len(df_exhaust.index)), labels=df_exhaust.index, ha='center', rotation=0, fontsize=10)

    # Add value labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

    plt.subplots_adjust(bottom=0.2)  # Increase space between bars to avoid overlap
    plt.show()

    # Plot F1-score
    df_f1 = pd.DataFrame({key: metrics_data['f1_score'][key] for key in types_exhaust_f1}).T
    ax = df_f1.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_title('F1-score by Type')
    ax.set_xlabel('Types')
    ax.set_ylabel('F1-score')
    plt.legend(title='Ontologies')
    plt.xticks(ticks=range(len(df_f1.index)), labels=df_f1.index, ha='center', rotation=0, fontsize=10)

    # Add value labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

    plt.subplots_adjust(bottom=0.2)  # Increase space between bars to avoid overlap
    plt.show()


def main():
    plot_combined_metrics()


if __name__ == "__main__":
    main()
