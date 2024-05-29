import csv
import os
import pickle
import time
import config
from typing import List, Literal
from matplotlib import pyplot as plt
from yaml import dump
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
    
    
def prepare_dataset_moons(samples=100):
    """
    Loads the moons dataset
    """
    moons = make_moons(n_samples=samples)
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
    plt.close()
    

def save_losses_weights_predictions(fname: str, losses: list[float], weights: list[list[float]], predictions: list[list[Literal[0,1]]]):
    with open(fname, 'w', newline="") as csvfile:
        field_names = ['loss', 'weights', 'predictions']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i in range(len(losses)):
            writer.writerow({'loss': losses[i], 'weights': weights[i], 'predictions': predictions[i]})


def save_classification_report(classification_report: dict, filename:str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_directory = os.path.join(script_dir, "classification_reports")
    full_path = (os.path.join(txt_directory, filename + ".txt"))
    # add configs
    classification_report["Parameters"] = {
        "RANDOM_SEED": config.RANDOM_SEED,
        "DEPTH": config.Q_DEPTH,
        "OPTIMIZER": config.OPTIM_METHOD,
        "SHOTS": config.N_SHOTS,
        "SAMPLES": config.SAMPLES,
        "TEST_FRAC": config.TEST_SIZE,
        "DATASET": config.DATASET_FUNCTION,
        "FEATURE_MAP": config.FEATURE_MAP
    }
    with open(full_path, "w") as txt_file:
        dump(classification_report, txt_file, sort_keys=False)


def save_weights_config_test_data(weights, test_data, test_labels, filename):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    file_location = os.path.join(checkpoint_dir, filename + ".pickle")
    config_dict = {"q_depth": config.Q_DEPTH, "n_shots": config.N_SHOTS, "n_samples": config.SAMPLES, "dataset_function": config.DATASET_FUNCTION}
    save_dict = {"weights": weights, "config": config_dict, "test_data": test_data, "test_labels": test_labels}
    with open(file_location, 'wb') as file:
        pickle.dump(save_dict, file)

    
if __name__ == "__main__":
    pass