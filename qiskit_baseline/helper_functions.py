import csv
import math
import os
import pickle
import time
from qiskit_baseline import config
from typing import Iterable, List, Literal
from matplotlib import pyplot as plt
from yaml import dump
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset_iris(n_features=2):
    """
    Loads the Iris dataset from scikit-learn, filters out the first two features from each sample
    and transforms it into a binary classification problem by ommitting one class.
    Then the features are normalized fit in the range [0,1]
    """
    # assert n_features is between 2 and 4
    if not (2 <= n_features <=4):
        raise ValueError("Only supports between 2 and 4 features per data point")
    iris = load_iris()
    # use only first two features of the iris dataset
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


def calculate_parity(counts: dict) -> list:
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


def calculate_interpret_result(counts: dict, weights: list[float]):
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
    bit_strings = []
    for i in range(0, int(math.pow(2, n_bits))):
        bit_strings.append(format(i, "0" + str(n_bits) + "b"))
    return bit_strings


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
        "FEATURE_MAP": config.FEATURE_MAP,
        "COMMENT": config.FILENAME_ADDON
    }
    with open(full_path, "w") as txt_file:
        dump(classification_report, txt_file, sort_keys=False)


def save_circuit(create_circuit_fun, filename):
    circuit = create_circuit_fun(config.N_QUBITS, config.Q_DEPTH, np.ones(config.N_QUBITS), np.zeros(config.N_QUBITS * (config.Q_DEPTH + 1)))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(script_dir, "circuits")
    full_path = (os.path.join(circuit_dir, filename + ".png"))
    print("Saving circuit to PNG")
    circuit.draw(output='mpl', filename = full_path)
    


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
        

def lower_bound_constraint_with_split(x: Iterable, params_index: int) -> float:
    thetas, interpret_weights = np.split(x, [params_index])
    thetas_constraint = thetas - config.LB_THETAS
    interpret_weights_constraint = interpret_weights - config.LB_INTERPRET
    return np.concatenate((thetas_constraint, interpret_weights_constraint))


def upper_bound_constraint_with_split(x: Iterable, params_index: int) -> float:
    thetas, interpret_weights = np.split(x, [params_index])
    thetas_constraint = config.UB_THETAS - thetas
    interpret_weights_constraint = config.UB_INTERPRET - interpret_weights
    return np.concatenate((thetas_constraint, interpret_weights_constraint))

    
if __name__ == "__main__":
    print(generate_bit_strings_of_n_bits(4))
