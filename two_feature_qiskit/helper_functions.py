import os
from typing import List, Literal
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset_iris():
    """
    Loads the Iris dataset from scikit-learn, filters out the first two features from each sample
    and transforms it into a binary classification problem by ommitting one class.
    Then the features are normalized fit in the range [0,1]
    """
    iris = load_iris()
    # use only first two features of the iris dataset
    X = iris.data[:,:2]
    # filter out only zero and one classes
    filter_mask = np.isin(iris.target, [0,1])
    X_filtered = X[filter_mask]
    y_filtered = iris.target[filter_mask]
    # min max scale features to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled, y_filtered
    
    
def prepare_dataset_moons():
    """
    Loads the moons dataset
    """
    moons = make_moons()
    X = moons[0]
    y = moons[1]
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def check_parity(qubits: List[int]) -> Literal[0,1]:
    """
    Checks the parity of a list of qubits.
    Returns 0 if there is an even number of 1s (or 0),
    Returns 1 if there is an odd number of 1s.
    
    :param: qubits: The qubit results
    :returns: 0 or 1
    """
    parity = sum(qubits) % 2
    return parity


def plot_losses(filename: str, losses) -> None:
    plt.plot(losses)
    plt.xlabel("iteration number")
    plt.ylabel("log loss")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))  
    

def plot_accuracy(filename: str, accuracy_scores) -> None:
    plt.plot(accuracy_scores)
    plt.xlabel("iteration number")
    plt.ylabel("accuracy score")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))  
    

def plot_acc_and_loss(filename: str, accuracy_scores: list[float], losses: list[float]):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))
    
    
if __name__ == "__main__":
    accs = [0.1, 0.2, 0.3, 0.4, 0.5]
    losses = [20.0, 15.0, 14.3, 14.5, 14.0]
    plot_acc_and_loss(None, accs, losses)