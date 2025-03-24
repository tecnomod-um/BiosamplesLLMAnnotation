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
        df = data_process('./results/contribution_file_CL.json')
        col_1 = 'CLO'
        col_2 = 'BTO'
    elif type == 'CT':
        df = data_process('./results/contribution_file_CT.json')
        col_1 = 'CL'
        col_2 = 'BTO'
    elif type == 'A':
        df = data_process('./results/contribution_file_A.json')
        col_1 = 'UBERON'
        col_2 = 'BTO'
    elif type == 'dash':
        return 0  # Set perfect match to 0 for 'dash'
    else:
        raise ValueError("Unrecognized ontology type")

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


def calculate_metrics(ontology_type):
    """
    Calculate precision, recall (exhaustiveness), and F1-score for each ontology type.

    Parameters:
        ontology_type (str): The ontology type ('CL', 'CT', 'A', or 'dash').

    Returns:
        tuple: Three dictionaries containing the accuracy (precision), recall (exhaustiveness),
               and F1-score for each ontology.
    """
    if ontology_type == 'CL':
        df = data_process('./results/contribution_file_CL.json')
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    elif ontology_type == 'CT':
        df = data_process('./results/contribution_file_CT.json')
        suffixes = ['CL', 'UBERON', 'BTO']
    elif ontology_type == 'A':
        df = data_process('./results/contribution_file_A.json')
        suffixes = ['UBERON', 'BTO']
    elif ontology_type == 'dash':
        df = df_dash
        suffixes = ['CLO', 'CL', 'UBERON', 'BTO']
    else:
        raise ValueError("Unrecognized ontology type")

    precisions = {}
    accuracies = {}
    recall = None if ontology_type == 'dash' else {}
    f1 = None if ontology_type == 'dash' else {}

    for suffix in suffixes:
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                true_val = df.iloc[i][true_col]
                pred_val = df.iloc[i][pred_col]
                # Skip rows where both true and predicted values are "-"
                if true_val == "-" and pred_val == "-":
                    #print(true_val, pred_val)
                    tn += 1
                elif true_val != "-" and pred_val == "-":
                    #print(true_val, pred_val)
                    fn += 1
                elif true_val == pred_val:
                    #print(true_val, pred_val)
                    tp += 1
                elif true_val != pred_val:
                    print(true_val,"|||",pred_val)
                    fp += 1
            print(ontology_type, suffix, "TP:", tp, " FP:", fp, " FN:", fn, " TN:", tn)
            # Calculate metrics if not in 'dash' mode
            if ontology_type != 'dash':
                # Calculate recall (exhaustiveness) and F1-score
                recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_value = (2 * tp) / (2 * tp + fn + fp) if ( 2 * tp + fn + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fn + fp)
                recall[suffix] = recall_value
                f1[suffix] = f1_value
                accuracies[suffix] = accuracy

            # Calculate accuracy (precision) based on the evaluated cases (excluding skipped rows)
            total_evaluated = tp + fp
            precision = tp / total_evaluated if total_evaluated > 0 else 0
            precisions[suffix] = precision

        else:
            if ontology_type == 'dash':
                recall[suffix] = None
                f1[suffix] = None
                precisions[suffix] = None
                accuracies[suffix] = None

    return precisions, recall, f1 , accuracies


def plot_combined_metrics():
    """
    Plot the calculated metrics for precision, exhaustiveness, and F1-score for all ontology types (CL, CT, A).
    """
    types = ['CL', 'CT', 'A', 'dash']
    metrics_data = {'precision': {}, 'accuracy': {} ,'exhaustiveness': {}, 'f1_score': {}, 'perfect_match': {}}

    for type in types:
        precision, exhaust, f1, accuracy = calculate_metrics(type)
        metrics_data['precision'][type] = precision
        metrics_data['accuracy'][type] = accuracy
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

    # Plot f1-score
    df_acc = pd.DataFrame({key: metrics_data['f1_score'][key] for key in types_exhaust_f1}).T
    ax = df_acc.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_title('F1-score by Type')
    ax.set_xlabel('Types')
    ax.set_ylabel('F1-score')
    plt.legend(title='Ontologies')
    plt.xticks(ticks=range(len(df_acc.index)), labels=df_acc.index, ha='center', rotation=0, fontsize=10)

    # Add value labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')

    plt.subplots_adjust(bottom=0.2)  # Increase space between bars to avoid overlap
    plt.show()

    # Plot accuracy
    types_exhaust_f1 = ['CL', 'CT', 'A']  # Remove 'dash' from exhaustiveness and F1-score plots
    df_exhaust = pd.DataFrame({key: metrics_data['accuracy'][key] for key in types_exhaust_f1}).T
    ax = df_exhaust.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_title('Accuracy by Type')
    ax.set_xlabel('Types')
    ax.set_ylabel('Accuracy')
    plt.legend(title='Ontologies',
               bbox_to_anchor=(1, 1))  # Move legend outside the plot to avoid overlap
    plt.xticks(ticks=range(len(df_exhaust.index)), labels=df_exhaust.index, ha='center', rotation=0, fontsize=10)

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
