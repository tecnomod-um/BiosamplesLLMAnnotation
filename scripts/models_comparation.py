import pandas as pd #dataframe manipulation
import matplotlib.pyplot as plt #data visualization

from df_comparation import df_comparation_gpt3_5, df_comparation_gpt4, df_comparation_gpt4o, df_comparation_ft, df_comparation_ft_4o

def get_accuracy(df):
    """
    Calculate the accuracy for each ontology in the given dataframe.

    Parameters:
        df (DataFrame): The dataframe containing predictions and true values
                         for each ontology. Columns are expected to include
                         pairs of columns for each ontology with the suffixes '_C'
                         (for true values) and '_M' (for model predictions).

    """
    for column in df:
        df[column] = df[column].fillna('unknown')
    
    suffixes = ['CLO', 'CL', 'UBERON', 'BTO']

    accuracies = {}
    for suffix in suffixes: #for each ontology the accuracy is calculated
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        true_pos=0
        false_pos = 0
        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                if df.iloc[i][true_col] == df.iloc[i][pred_col]:
                    true_pos += 1
                else : 
                    false_pos +=1
            accuracy= true_pos/(true_pos+false_pos)
            accuracies[suffix] = accuracy
        else:
            accuracies[suffix] = None
    return accuracies

def plot_accuracies(models_data, model_names):
    """
    Plot the accuracies of different models for each ontology.

    Parameters:
        models_data (list of DataFrame): List of dataframes, each containing
                                          prediction and true value columns for
                                          a model and each ontology.
        model_names (list of str): List of model names corresponding to each dataframe
                                 in models_data, used as labels in the plot.
    """
    # Function to plot the accuracies of the models
    accuracies = [get_accuracy(df) for df in models_data]
    ontologies = ['CLO', 'CL', 'UBERON', 'BTO']
    data = {ont: [acc[ont] for acc in accuracies] for ont in ontologies}
    df_plot = pd.DataFrame(data, index=model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the width of each bar and the space between groups
    bar_width = 4
    spacing = 5
    positions = [i * (bar_width * len(ontologies) + spacing) for i in range(len(models_data))]

    for i, ont in enumerate(ontologies):
        bar_positions = [pos + i * bar_width for pos in positions]
        ax.bar(bar_positions, df_plot[ont], width=bar_width, label=ont)

        # Add values on top of each bar
        for j, pos in enumerate(bar_positions):
            ax.annotate(f'{df_plot[ont][j]:.3f}', 
                        (pos, df_plot[ont][j]), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points')

    # Labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Precision')
    ax.set_title('Ontology Precision by Model')
    ax.set_ylim(0, 1)  # Set y-axis limit to 1
    ax.set_xticks([pos + (bar_width * (len(ontologies) - 1)) / 2 for pos in positions])
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.show()

def main():
    models_data = [df_comparation_gpt3_5, df_comparation_gpt4, df_comparation_gpt4o, df_comparation_ft,  df_comparation_ft_4o]
    model_names = ['Model GPT-3.5', 'Model GPT-4', 'Model GPT-4o', 'Ft GPT-3.5 Model ', 'Ft GPT-4o-mini Model']
    plot_accuracies(models_data, model_names)

if __name__ == "__main__":
    main()
