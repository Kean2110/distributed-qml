import functools
import glob
import os
import pickle
import shutil
import time
from typing import Iterable, List, Tuple, Union, Literal
from netqasm.sdk import Qubit, EPRSocket
from sklearn.datasets import make_moons, load_iris
import numpy as np
from numpy.typing import NDArray
import random
import math
from sklearn.preprocessing import MinMaxScaler
from utils.logger import logger
import utils.constants

from yaml import dump


def remove_folder_except(folder_path: str, except_paths: list[str]):
    for item in os.listdir(folder_path):
        if item not in except_paths:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                

def generate_chunks_with_max_size(max_size: int, depth: int) -> list[int]:
    output_length = math.ceil(max_size / depth)
    no_of_full_chunks = depth // max_size
    output_chunks = [max_size] * no_of_full_chunks
    if depth % max_size != 0:
        output_chunks.append(depth % max_size)
    return output_chunks                    


def split_array_by_nth_occurrences(n: int, array, element) -> np.array:
    '''
    Takes an array, an element and a number as an input.
    This function splits the input array among every n-th occurrence of the element.
    So in every split array there is exactly n occurrences of the element.
    Example:
        `` split_array_by_nth_occurrences(2, [0,0,1,1,0,0,1,1,0,0,0], 1)
        `` [[0,0,1,1], [0,0,1,1,0,0,0]]
    '''
    nth_indices = []
    n_occurrences = 0
    for i, elem in enumerate(array):
        if elem == element:
            n_occurrences += 1
        if n_occurrences % n == 0 and n_occurrences != 0:
            nth_indices.append(i + 1)
    return np.split(array, nth_indices)


def calculate_parity_from_shots(result_array_client1: list[Literal[0,1]], result_array_client2: list[Literal[0,1]]) -> Literal[0,1]:
    n_shots = len(result_array_client1)
    str_arr_combined_results = [str(result_array_client1[i])+str(result_array_client2[i]) for i in range(n_shots)]
    zeros = 0
    ones = 0
    for single_combined_measurement in str_arr_combined_results:
        if single_combined_measurement.count('1') % 2 == 0:
            zeros += 1
        else:
            ones += 1
    output_probs = [zeros / n_shots, ones / n_shots]
    return output_probs.index(max(output_probs))


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


def split_data_into_batches(data: List[Tuple[float, float]], labels: List[int], batch_size: int, random_seed: int = 42) -> Tuple[List[List[Tuple[float,float]]],List[int]]:
    """
    Splits the data into batches of size batch_size.
    If the data is not dividable by batch_size, the last batch is smaller.
    
    :param: batch_size: the size of each batch
    :data: the data samples as a list
    :returns: the batches that were created
    """
    # randomly shuffles the data
    combined_data = list(zip(data, labels))
    random.seed(random_seed)
    random.shuffle(combined_data)
    shuffled_data, shuffled_labels = zip(*combined_data)
    shuffled_data, shuffled_labels = np.array(shuffled_data), np.array(shuffled_labels)
    
    # if the number of samples is dividable by the batch size
    if len(data) % batch_size == 0:
        n_batches = len(data) // batch_size
        data_batches = np.split(shuffled_data, n_batches)
        label_batches = np.split(shuffled_labels, n_batches)
        return data_batches, label_batches
    # calculate the number of full batches (that all have batch_size elements)
    n_full_batches = len(data) // batch_size
    # take the data points that make up the full batches
    full_batch_data = shuffled_data[:n_full_batches*batch_size]
    full_labels = shuffled_labels[:n_full_batches*batch_size]
    # using split and not array_split, so an exception is raised in case len(full_batch_data) % batch_size != 0
    first_batches = np.split(full_batch_data, n_full_batches)
    first_labels = np.split(full_labels, n_full_batches)
    # calculate the last batch
    last_batch = shuffled_data[n_full_batches*batch_size:]
    last_labels = shuffled_labels[n_full_batches*batch_size:]
    # append last batch to the first batches
    first_batches.append(last_batch)
    first_labels.append(last_labels)
    return first_batches, first_labels


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
    scaler = MinMaxScaler(feature_range=(utils.constants.LOWER_BOUND_INPUTS, utils.constants.UPPER_BOUND_INPUTS))
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled, y_filtered
    
    
def prepare_dataset_moons(n_samples: int = 100) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Loads the moons dataset
    """
    moons = make_moons(n_samples=n_samples)
    X = moons[0]
    y = moons[1]
    scaler = MinMaxScaler(feature_range=(utils.constants.LOWER_BOUND_INPUTS, utils.constants.UPPER_BOUND_INPUTS))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def phase_gate(angle: float, qubit: Qubit):
    #qubit * np.exp((1j * angle)/2)
    qubit.rot_Z(angle=angle)
 
 
def save_classification_report(filename: str, output_dir: str, classification_report: dict):
    txt_directory = os.path.join(output_dir, "classification_reports")
    if not os.path.exists(txt_directory):
        os.mkdir(txt_directory)
    full_path = (os.path.join(txt_directory, filename + ".txt"))
    with open(full_path, "w") as txt_file:
        dump(classification_report, txt_file, sort_keys=False)


def load_latest_checkpoint(checkpoint_dir: str):
    # List all pickle files that are in the checkpoint dir
    checkpoints = [f for f in glob.glob(os.path.join(checkpoint_dir, "*.pickle"))]
    try:
        # Get the latest checkpoint pickle file
        latest_checkpoint = max(checkpoints, key = lambda x: os.path.getmtime(x))
        with open(latest_checkpoint, 'rb') as file:
            # load the data and return the params, last iteration, losses and accs
            data = pickle.load(file)
            return data["params"], data["iter"], data["losses"], data["accs"]
    except (ValueError, FileNotFoundError, KeyError) as e:
        logger.warning("No checkpoint found, returning None")
        # return None and 0 if no file is found
        return None, 0, [], []
    
    
def load_latest_input_checkpoint(checkpoint_dir: str):
    # List all pickle files that are in the checkpoint dir
    checkpoints = [f for f in glob.glob(os.path.join(checkpoint_dir, "*.pickle"))]
    try:
        # Get the latest checkpoint pickle file
        latest_checkpoint = max(checkpoints, key = lambda x: os.path.getmtime(x))
        with open(latest_checkpoint, 'rb') as file:
            # load the data and return the params and the last iteration
            data = pickle.load(file)
            logger.info("Loaded weights and config from input checkpoint")
            return data
    except (ValueError, FileNotFoundError) as e:
        logger.warning("No checkpoint found")
        raise FileNotFoundError(f"No files found in {checkpoint_dir}")
    
    
def lower_bound_constraint(x: Iterable):
    return x - utils.constants.LOWER_BOUND_PARAMS


def upper_bound_constraint(x: Iterable):
    return utils.constants.UPPER_BOUND_PARAMS - x
        


if __name__ == '__main__':
    #print(generate_chunks_with_max_size(4, 5))
    n = 4
    array = [0,1,1,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0]
    print(split_array_by_nth_occurrences(4, array, 1))
    
    
    