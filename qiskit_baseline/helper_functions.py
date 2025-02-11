import csv
import math
import os
import pickle
import time
from qiskit_baseline import config
from typing import Callable, Iterable, List, Literal, Tuple
from matplotlib import pyplot as plt
from yaml import dump
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset_iris(n_features:int = 2) -> Tuple[list, list]:
    """
    Loads the Iris dataset from scikit-learn, filters out the first two features from each sample
    and transforms it into a binary classification problem by ommitting one class.
    Then the features are normalized fit in the range [0,1].
    
    :param n_features: Number of features.
    :returns: Data points and their labels.
    """
    # assert n_features is between 2 and 4
    if not (2 <= n_features <=4):
        raise ValueError("Only supports between 2 and 4 features per data point")
    iris = load_iris()
    # use only first n features of the iris dataset
    X = iris.data[:,:n_features]
    # filter out only zero and one classes
    filter_mask = np.isin(iris.target, [0,1])
    X_filtered = X[filter_mask]
    y_filtered = iris.target[filter_mask]
    # min max scale features to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler = MinMaxScaler(feature_range=(0, 2*math.pi))
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled, y_filtered
    
    
def prepare_dataset_moons(n_samples:int = 100) -> Tuple[list, list]:
    """
    Loads the moons dataset and scales it to (0,1)
    :param n_samples: Number of samples
    :returns: Data points and their labels
    """
    moons = make_moons(n_samples=n_samples)
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


def calculate_parity(counts: dict) -> list:
    """
    Determines the class label based on measurement counts.
    The class is assigned based on parity (even or odd number of '1's).
    
    :param counts: Dictionary mapping bitstrings to their occurrence count.
    :returns: The predicted class label (0 or 1).
    """
    zeros = 0
    ones = 0
    for measure, count in counts.items():
        if measure.count('1') % 2 == 0:
            zeros += count
        else:
            ones += count
    output_probs = [zeros / config.N_SHOTS, ones / config.N_SHOTS]
    # return class
    return output_probs.index(max(output_probs))


def calculate_expectation_value(counts: dict) -> list:
    """
    Computes the expectation value of the measurement results.
    
    :param counts: Dictionary mapping bitstrings to their occurrence count.
    :returns: Probability of belonging to class 1.
    """
    zeros = 0
    ones = 0
    for measure, count in counts.items():
        if measure.count('1') % 2 == 0:
            zeros += count
        else:
            ones += count
    output_probs = [zeros / config.N_SHOTS, ones / config.N_SHOTS]
    # return prob that it belongs to class 1
    return output_probs[1]


def calculate_interpret_result(counts: dict, weights: list[float]) -> int:
    """
    Interprets the quantum measurement results either by using weights or throguh the parity.
    Interpret function derived from: https://arxiv.org/abs/2106.12819
    
    :param counts: Dictionary of measurement counts.
    :param weights: Optional weight values.
    :returns: Class prediction (0 or 1).
    """
    possible_bit_strings = generate_bit_strings_of_n_bits(config.N_QUBITS)
    interpret_sum = 0
    for i, bit_string in enumerate(possible_bit_strings):
        if bit_string in counts:
            prob = counts[bit_string] / config.N_SHOTS
            if len(weights) > 0:
                interpret_sum += prob * weights[i]
            else:
                # if no weights are specified we just use the parity
                parity = bit_string.count('1') % 2
                parity_sign = parity if parity == 1 else -1
                interpret_sum += prob * parity_sign
    return 0 if interpret_sum <= 0 else 1


def generate_bit_strings_of_n_bits(n_bits: int) -> list[str]:
    """
    Generates all possible bitstrings of a given length.
    
    :param n_bits: Number of qubits.
    :returns: List of bitstrings.
    """
    bit_strings = []
    for i in range(0, int(math.pow(2, n_bits))):
        bit_strings.append(format(i, "0" + str(n_bits) + "b"))
    return bit_strings


def plot_losses(filename: str, losses) -> None:
    """
    Plots and saves the loss curve.
    
    :param filename: Output filename.
    :param losses: Loss values.
    """
    plt.plot(losses)
    plt.xlabel("iteration number")
    plt.ylabel("log loss")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))  
    

def plot_accuracy(filename: str, accuracy_scores) -> None:
    """
    Plots and saves accuracy scores.
    
    :param filename: Output filename.
    :param accuracy_scores: Accuracy values.
    """
    plt.plot(accuracy_scores)
    plt.xlabel("iteration number")
    plt.ylabel("accuracy score")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))  
    

def plot_acc_and_loss(filename: str, accuracy_scores: list[float], losses: list[float]) -> None:
    """
    Plots and saves accuracy and loss curves in a single figure.
    
    :param filename: Output filename.
    :param accuracy_scores: List of accuracy scores.
    :param losses: List of log loss values.
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
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_directory = os.path.join(script_dir, "plots")
    plt.savefig(os.path.join(plot_directory, filename))
    plt.close()
    

def save_losses_weights_predictions(fname: str, losses: list[float], weights: list[list[float]], predictions: list[list[Literal[0,1]]]) -> None:
    """
    Saves losses, weights, and predictions to a CSV file for debugging purposes.
    
    :param fname: Filename for saving data.
    :param losses: List of loss values.
    :param weights: List of model weight values.
    :param predictions: List of predicted labels (0 or 1).
    """
    with open(fname, 'w', newline="") as csvfile:
        field_names = ['loss', 'weights', 'predictions']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i in range(len(losses)):
            writer.writerow({'loss': losses[i], 'weights': weights[i], 'predictions': predictions[i]})


def save_classification_report(classification_report: dict, filename:str) -> None:
    """
    Saves the classification report along with model configuration parameters.
    
    :param classification_report: Dictionary containing classification metrics.
    :param filename: Filename to save the report.
    """
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
        "FEATURE_MAP": config.FEATURE_MAP,
        "COMMENT": config.FILENAME_ADDON
    }
    with open(full_path, "w") as txt_file:
        dump(classification_report, txt_file, sort_keys=False)


def save_circuit(create_circuit_fun: Callable, filename: str) -> None:
    """
    Saves a quantum circuit visualization as an image.
    
    :param create_circuit_fun: Function that generates a quantum circuit.
    :param filename: Output filename for the circuit image.
    """
    circuit = create_circuit_fun(config.N_QUBITS, config.Q_DEPTH, np.ones(config.N_QUBITS), np.zeros(config.N_QUBITS * (config.Q_DEPTH + 1)))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(script_dir, "circuits")
    full_path = (os.path.join(circuit_dir, filename + ".png"))
    print("Saving circuit to PNG")
    circuit.draw(output='mpl', filename = full_path)
    

def save_weights_config_test_data_losses_accs(weights: list[float], test_data: list[float], test_labels: list[Literal[0,1]], losses: list[float], accs: list[float], filename: str) -> None:
    """
    Saves model weights, test data, labels, losses, and accuracy scores as a pickle file.
    
    :param weights: Model weights.
    :param test_data: Test dataset.
    :param test_labels: Corresponding test labels.
    :param losses: Loss values recorded during training.
    :param accs: Accuracy scores.
    :param filename: Filename for the saved checkpoint.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(curr_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    file_location = os.path.join(checkpoint_dir, filename + ".pickle")
    config_dict = {"q_depth": config.Q_DEPTH, "n_shots": config.N_SHOTS, "n_samples": config.SAMPLES, "dataset_function": config.DATASET_FUNCTION}
    save_dict = {"weights": weights, "config": config_dict, "test_data": test_data, "test_labels": test_labels, "losses": losses, "accs": accs}
    with open(file_location, 'wb') as file:
        pickle.dump(save_dict, file)
        

def lower_bound_constraint_with_split(x: Iterable, params_index: int) -> np.ndarray:
    """
    Enforces lower bound constraints on parameter values, 
    while accounting that the parameters X must be split when using the interpret function for parity calculation.
    
    :param x: Iterable containing parameter values.
    :param params_index: Index to split parameters.
    :returns: Constraint values as a NumPy array.
    """
    thetas, interpret_weights = np.split(x, [params_index])
    thetas_constraint = thetas - config.LB_THETAS
    interpret_weights_constraint = interpret_weights - config.LB_INTERPRET
    return np.concatenate((thetas_constraint, interpret_weights_constraint))


def upper_bound_constraint_with_split(x: Iterable, params_index: int) -> np.ndarray:
    """
    Enforces upper bound constraints on parameter values,
    while accounting that the parameters X must be split when using the interpret function for parity calculation.
    
    :param x: Iterable containing parameter values.
    :param params_index: Index to split parameters.
    :returns: Constraint values as a NumPy array.
    """
    thetas, interpret_weights = np.split(x, [params_index])
    thetas_constraint = config.UB_THETAS - thetas
    interpret_weights_constraint = config.UB_INTERPRET - interpret_weights
    return np.concatenate((thetas_constraint, interpret_weights_constraint))

    
if __name__ == "__main__":
    # only for testing purposes
    print(generate_bit_strings_of_n_bits(4))
