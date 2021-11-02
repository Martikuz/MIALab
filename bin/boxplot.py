import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Local variables for plotting
AVAILABLE_METRICS = ['DICE', 'HDRFDST95']   # mk: added code that renames HDRFDST to HDRFDST95

def read_pure_python(dir_path: str):
    # Get the correct file
    csv_files = [f.path for f in os.scandir(dir_path) if f.name == 'results.csv']
    if len(csv_files) != 1:
        raise ValueError('No results.csv file found in the specified directory!')
    csv_file = csv_files[0]

    # Read the content of the file and the header (column names)
    with open(csv_file, 'r') as file:
        file_data = [line.replace('\n', '').split(';') for line in file.readlines()]
    return file_data[1:], file_data[0]

def read_with_pandas(dir_path: str):
    # Get the correct file
    csv_files = [f.path for f in os.scandir(dir_path) if f.name == 'results.csv']
    if len(csv_files) != 1:
        raise ValueError('No results.csv file found in the specified directory!')
    csv_file = csv_files[0]

    # Read the content of the file and the header (column names)
    data = pd.read_csv(csv_file, delimiter=';')
    return data.values.tolist(), data.columns.to_list()

def prepare_data_for_plotting(header: list, data: list, metrics: tuple = ('DICE',)):
    # Get the unique labels
    idx_structure = header.index('LABEL')
    unique_labels = list(set([entry[idx_structure] for entry in data]))
    output_data = list()

    # Get the correct metric identifier
    for metric in metrics:
        if metric not in header:
            raise ValueError('One of the selected metrics is not contained in the header!')
        idx_metric = header.index(metric)

        # Collect the appropriate metric values for each label
        # (We do not need the patient information anymore for plotting)
        formatted_data = {k: list() for k in unique_labels}
        [formatted_data[entry[idx_structure]].append(float(entry[idx_metric])) for entry in data]
        output_data.append(formatted_data)
    return output_data


def set_boxplot_format(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['caps'], linewidth=0)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=1.5)
    plt.setp(bp['fliers'], marker='.')
    plt.setp(bp['fliers'], markerfacecolor='black')
    plt.setp(bp['fliers'], alpha=1)


def generate_boxplot(data: list, label: str, output_directory: str, timestamp: str, title: str, y_label: str,
                     metric: str, min_: float = None, max_: float = None):

    # Prepare the data for plotting again
    if not all([True if label in entry['data'].keys() else False for entry in data]):
        raise ValueError(f'The selected label ({label}) is not found for all experiments!')
    plot_data = [entry['data'][label] for entry in data]
    x_ticks = [entry['experiment'] for entry in data]

    # Generate the plot structure
    fig = plt.figure(figsize=(7.2, 9.6))  # figsize defaults to (width, height) = (6.4, 4.8)
    ax = fig.add_subplot(111)  # create an axes instance (nrows = ncols = index)
    bp = ax.boxplot(plot_data, widths=0.6)
    set_boxplot_format(bp, '000')

    # Set and format title, labels, and ticks
    font_dict = {'fontsize': 12, 'fontweight': 'bold'}
    ax.set_title(title, fontdict=font_dict)
    ax.set_ylabel(y_label, fontdict=font_dict)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticklabels(x_ticks, fontdict=font_dict)
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Adjust min and max if provided
    if min_ is not None or max_ is not None:
        min_original, max_original = ax.get_ylim()
        min_ = min_ if min_ is not None and min_ < min_original else min_original
        max_ = max_ if max_ is not None and max_ > max_original else max_original
        ax.set_ylim(min_, max_)

    file_path = os.path.join(output_directory, f'plot_{timestamp}_{label}_{metric}.png')
    plt.savefig(file_path)
    plt.close()

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    #pass  # pass is just a placeholder if there is no other code

    # Configuration
    metric_to_compute = ('DICE', 'HDRFDST95')   # mk: added code that renames HDRFDST to HDRFDST95
    structures_to_compute = ('Thalamus',)
    experiment_paths = ('E:/01_Programme/Git/00_MyFolder/MIALab/bin/mia-result/2021', 'E:/01_Programme/Git/00_MyFolder/MIALab/bin/mia-result/2021')
    experiment_names = ('First Experiment', 'Second Experiment')
    plot_output_directory = './mia-result/'
    plot_title = ('Comparison of Experiments on Thalamus',)
    plot_ylabels = ('Dice', 'Hausdorff-Distance')
    plot_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Check the configuration
    if len(experiment_names) != len(experiment_paths):
        raise ValueError('The number of experiments names is not equal to the number of experiment paths! '
                         'Please check configuration...')
    for metric in metric_to_compute:
        if metric not in AVAILABLE_METRICS:
            raise ValueError('One of the metrics to compute is not available! Please check the configuration...')
    if not os.path.exists(plot_output_directory) and not os.path.isdir(plot_output_directory):
        os.mkdir(plot_output_directory)

    # Read the data and prepare it for plotting
    experiment_data = list()
    for path, name in zip(experiment_paths, experiment_names):
        # (Un-)comment the reading method
        data, header = read_with_pandas(path)
        # data, header = read_pure_python(path)

        # Prepare and add the data to the experment_data
        formatted_data = prepare_data_for_plotting(header, data, metric_to_compute)
        for metric, data in zip(metric_to_compute, formatted_data):
            experiment_data.append({'experiment': name, 'header': header, 'metric': metric, 'data': data})

    # Plot the data
    for metric, ylabel in zip(metric_to_compute, plot_ylabels):
        data = [entry for entry in experiment_data if entry['metric'] == metric]
        for structure, title in zip(structures_to_compute, plot_title):
            if metric == 'DICE':
                generate_boxplot(data, structure, plot_output_directory, plot_timestamp, title, ylabel, metric,
                                 min_=0, max_=1)
            else:
                generate_boxplot(data, structure, plot_output_directory, plot_timestamp, title, ylabel, metric)


if __name__ == '__main__':
    main()
