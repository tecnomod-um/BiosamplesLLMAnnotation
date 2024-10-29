import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def add_trendline(ax, x, y, color, label):
    """
    Adds a trendline to a plot.

    Parameters:
    - ax: The axis on which to plot the trendline.
    - x: The x-values for the trendline.
    - y: The y-values for the trendline.
    - color: Color of the trendline.
    - label: Label for the trendline in the legend.
    """
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle='dashed', color=color, label=f'Trendline {label}')

def plot_training_accuracy(ax, data, title):
    """
    Plots training accuracy.

    Parameters:
    - ax: The axis on which to plot.
    - data: Data containing the 'train_accuracy' column.
    - title: Title of the plot.
    """
    ax.plot(data[data['train_accuracy'].notnull()]['train_accuracy'])
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Train Accuracy')

def plot_validation_loss(ax, data, title):
    """
    Plots validation loss with a trendline.

    Parameters:
    - ax: The axis on which to plot.
    - data: Data containing 'step' and 'valid_loss' columns.
    - title: Title of the plot.
    """
    ax.scatter(data['step'], data['valid_loss'], label='Validation Loss', color='red')
    add_trendline(ax, data['step'], data['valid_loss'], color='red', label='Validation Loss')
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()

def plot_training_vs_validation_loss(ax, data, title):
    """
    Plots training loss and validation loss with a trendline for validation.

    Parameters:
    - ax: The axis on which to plot.
    - data: Data containing 'step', 'train_loss', and 'valid_loss' columns.
    - title: Title of the plot.
    """
    ax.plot(data['step'], data['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    ax.scatter(data['step'], data['valid_loss'], label='Validation Loss', color='red', s=40)
    add_trendline(ax, data['step'], data['valid_loss'], color='red', label='Validation Trendline')
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend()

def main():
    """
    Main function to load data, create plots, and display them.
    """
    # Load CSV files
    file_path1 = 'result_35.csv'  # First file path
    file_path2 = 'result_4o.csv'  # Second file path for comparison
    data1 = pd.read_csv(file_path1)
    data2 = pd.read_csv(file_path2)

    # Create a 3x2 grid of plots (one column for each file)
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    # First row: Training Accuracy
    plot_training_accuracy(axs[0, 0], data1, 'Training Accuracy (GPT-3.5-turbo)')
    plot_training_accuracy(axs[0, 1], data2, 'Training Accuracy (GPT-4o-mini)')

    # Second row: Validation Loss
    plot_validation_loss(axs[1, 0], data1, 'Validation Loss (GPT-3.5-turbo)')
    plot_validation_loss(axs[1, 1], data2, 'Validation Loss (GPT-4o-mini)')

    # Third row: Training vs Validation Loss
    plot_training_vs_validation_loss(axs[2, 0], data1, 'Training vs Validation Loss (GPT-3.5-turbo)')
    plot_training_vs_validation_loss(axs[2, 1], data2, 'Training vs Validation Loss (GPT-4o-mini)')

    # Display all plots
    plt.show()

if __name__ == "__main__":
    main()
