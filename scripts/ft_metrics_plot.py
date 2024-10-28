import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def add_trendline(ax, x, y, color, label):
    """
    Function to add a trendline to a plot based on the given x and y data points.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to plot the trendline on.
        x (array-like): X-axis data points.
        y (array-like): Y-axis data points.
        color (str): Color of the trendline.
        label (str): Label for the trendline in the plot legend.
    """
    mask = ~np.isnan(y) # Filter out null values
    x = x[mask]
    y = y[mask]
    if len(x) > 1:  # Ensure there are enough points to calculate the trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle='dashed', color=color, label=f'Trendline {label}') # Plot the trendline with a dashed style


def plot_metrics(file_path):
    """
    Function to load and visualize training and validation metrics from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing the metrics.
    """
    data = pd.read_csv(file_path)

    # First plot: Training Accuracy
    plt.figure(figsize=(10, 6))
    data[data['train_accuracy'].notnull()]['train_accuracy'].plot()
    plt.title('Training Accuracy')
    plt.xlabel('Index')
    plt.ylabel('Train Accuracy')
    plt.show()

    # Second plot: Validation Metrics
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot validation loss as points
    ax.scatter(data['step'], data['valid_loss'], label='Validation Loss', color='red')
    add_trendline(ax, data['step'], data['valid_loss'], color='red', label='Validation Loss')

    # Plot validation mean token accuracy as points
    ax.scatter(data['step'], data['valid_mean_token_accuracy'], label='Validation Mean Token Accuracy', color='orange')
    add_trendline(ax, data['step'], data['valid_mean_token_accuracy'], color='orange',
                  label='Validation Mean Token Accuracy')

    # Add titles and labels
    ax.set_title('Validation Metrics over Steps')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Values')
    ax.legend()

    # Show the plot
    plt.show()

    # Third plot: Training Loss vs Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training loss with transparency
    ax.plot(data['step'], data['train_loss'], label='Training Loss', color='blue', alpha=0.7, zorder=1)

    # Plot validation loss as larger points
    ax.scatter(data['step'], data['valid_loss'], label='Validation Loss', color='red', s=40, zorder=5)

    # Add the trendline
    add_trendline(ax, data['step'], data['valid_loss'], color='red', label='Validation Trendline')

    # Add titles and labels
    plt.title('Training Loss and Validation Loss over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.legend()

    # Show the plot
    plt.show()

def main():
    file_path_35 = 'result_35.csv'
    file_path_4o = 'result_4o.csv'
    plot_metrics(file_path_35)
    plot_metrics(file_path_4o)
if __name__ == "__main__":
    main()