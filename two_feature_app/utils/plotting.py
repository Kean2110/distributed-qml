from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def plot_losses(filename: str, losses: list[float]) -> None:
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("log loss")
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        plot_directory = os.path.join(parent_dir, "plots")
        plt.savefig(os.path.join(plot_directory, "loss_" + filename))
    
def plot_accs(filename: str, accuracy_scores: list[float]) -> None:
        plt.plot(accuracy_scores)
        plt.xlabel("iteration")
        plt.ylabel("accuracy score")
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        plot_directory = os.path.join(parent_dir, "plots")
        plt.savefig(os.path.join(plot_directory, "acc_" + filename))
        

def plot_accs_and_losses(filename: str, accuracy_scores: list[float], losses: list[float]):
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
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    plot_directory = os.path.join(parent_dir, "plots")
    plt.savefig(os.path.join(plot_directory, "acc_loss_" + filename))