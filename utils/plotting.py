from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np

def plot_losses(filename: str, output_dir: str, losses: list[float]) -> None:
        """
        Plots and saves the log loss values over iterations.

        :param filename: The name of the output file.
        :param output_dir: The directory where the plot will be saved.
        :param losses: A list of log loss values recorded over iterations.
        """
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("log loss")
        plot_directory = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_directory):
                os.mkdir(plot_directory)
        plt.savefig(os.path.join(plot_directory, "loss_" + filename))
    
def plot_accs(filename: str, output_dir: str, accuracy_scores: list[float]) -> None:
        """
        Plots and saves the accuracy scores over iterations.

        :param filename: The name of the output file.
        :param output_dir: The directory where the plot will be saved.
        :param accuracy_scores: A list of accuracy scores recorded over iterations.
        """
        plt.plot(accuracy_scores)
        plt.xlabel("iteration")
        plt.ylabel("accuracy score")
        plot_directory = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_directory):
                os.mkdir(plot_directory)
        plt.savefig(os.path.join(plot_directory, "acc_" + filename))
        

def plot_accs_and_losses(filename: str, output_dir: str, accuracy_scores: list[float], losses: list[float]) -> None:
        """
        Plots and saves accuracy scores and log loss values on the same figure with dual y-axes.

        :param filename: The name of the output file.
        :param output_dir: The directory where the plot will be saved.
        :param accuracy_scores: A list of accuracy scores recorded over iterations.
        :param losses: A list of log loss values recorded over iterations.
        """
        fig, ax1 = plt.subplots()
        color_losses = 'tab:blue'
        ax1.set_xlabel('iteration number')
        ax1.set_ylabel('log loss', color=color_losses)
        ax1.plot(losses, color=color_losses)
        ax1.tick_params(axis='y', labelcolor=color_losses)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_accs = 'tab:red'
        ax2.set_ylabel('accuracy score', color=color_accs)  # we already handled the x-label with ax1
        ax2.plot(accuracy_scores, color=color_accs)
        ax2.tick_params(axis='y', labelcolor=color_accs)
        fig.suptitle(filename)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plot_directory = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_directory):
                os.mkdir(plot_directory)
        plt.savefig(os.path.join(plot_directory, "acc_loss_" + filename))
    
    
def plot_data_with_moving_average(filename: str, output_dir: str, data: list[float], window_size: int):
        """
        Plots data with a moving average window.

        :param filename: The name of the output file.
        :param output_dir: The directory where the plot will be saved.
        :param data: A list of numerical data points to be plotted.
        :param window_size: The size of the moving average window.
        :returns: None
        """
        output_file = os.path.join(output_dir, filename)
        plt.clf()
        window = np.ones(window_size) / float(window_size)
        data_padded = np.pad(data, window_size//2, mode="edge")
        cumsum_vec = np.cumsum(data_padded)
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        plt.figure(figsize=(15,5))
        plt.plot(data, color ='blue', label='data', alpha=0.3)
        plt.plot(moving_avg, color='red', label=f'{window_size}-Point Moving Average')
        plt.legend()
        plt.savefig(output_file)
