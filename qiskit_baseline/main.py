from ast import Tuple
import math
from typing import Literal
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, BasicAer, execute
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from qiskit_baseline.helper_functions import calculate_parity, calculate_interpret_result, check_parity, lower_bound_constraint_with_split, plot_acc_and_loss, prepare_dataset_iris, prepare_dataset_moons, plot_accuracy, plot_losses, save_circuit, save_losses_weights_predictions, save_classification_report, save_weights_config_test_data_losses_accs, upper_bound_constraint_with_split, calculate_expectation_value
from scipy.optimize import minimize, Bounds
from utils.timer import global_timer
import numpy as np
from qiskit_baseline import config


def create_ZZ_feature_map(n_qubits: int, features: list[float]) -> QuantumCircuit:
    """
    Creates a ZZ feature map for quantum circuits.
    
    :param n_qubits: Number of qubits.
    :param features: Feature vector to encode.
    :returns: QuantumCircuit with ZZ feature mapping.
    """
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    for i, qbit in enumerate(qreg):
        circuit.h(qbit)
        circuit.p(2.0*features[i], qbit)
    for i, qbit in enumerate(qreg):
        if i < len(qreg) - 1:
            circuit.cx(qbit, qreg[i+1])
            circuit.p(2*(math.pi - features[i])*(math.pi - features[i+1]), qreg[i+1])
            circuit.cx(qbit, qreg[i+1])
    return circuit


def create_rot_feature_map(n_qubits: int, features: list[float]) -> QuantumCircuit:
    """
    Creates a rotational feature map for quantum circuits.
    
    :param n_qubits: Number of qubits.
    :param features: Feature vector to encode.
    :returns: QuantumCircuit with rotational feature mapping.
    """
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    for i, qubit in enumerate(qreg):
        circuit.ry(features[i], qubit)
    return circuit


def create_amplitude_encoding_feature_map(n_qubits: int, features: list[float]) -> QuantumCircuit:
    """
    Creates an amplitude encoding feature map for quantum circuits.
    
    :param n_qubits: Number of qubits.
    :param features: Feature vector to encode.
    :returns: QuantumCircuit with amplitude encoding feature mapping.
    """
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    initial_state = (1 / np.linalg.norm(features)) * features
    circuit.initialize(initial_state, range(n_qubits)) 
    return circuit       
            

def create_circuit(n_qubits: int, q_depth: int, features: list[float], weights: list[float]) -> QuantumCircuit:
    """
    Constructs a quantum circuit for classification by assembling feature map and constructing the parametrized layers.
    
    :param n_qubits: Number of qubits.
    :param q_depth: Number of layers.
    :param features: Feature vector to encode.
    :param weights: Trainable parameters.
    :returns: QuantumCircuit object.
    """
    assert len(weights) == n_qubits * (q_depth+1), "Number of weights doesn't match n_qubits * (q_depth+1)"
    # Specify register
    qreg_q = QuantumRegister(n_qubits, 'q')
    creg = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qreg_q, creg)
    
    feature_map = create_rot_feature_map(n_qubits, features)
    circuit = circuit.compose(feature_map)
    # adapt manually according to which remote CNOTs we leave out, if range(q_depth), we leave no RCNOTs out
    layers_with_rcnot = range(q_depth)
    # apply weights // entanglement layer
    for i in range(q_depth):
        for j, qbit in enumerate(qreg_q):
            circuit.ry(weights[i * n_qubits + j], qbit)
        for j, qbit in enumerate(qreg_q):
            # j <= len(qreg) - 1 means we leave out the last qubit
            if j < len(qreg_q) - 1:
                # j == 1 is the RCNOT, therefore we construct all CNOTs that are not remote (j != 1) --> only applicable for 4 QUBITS!
                # i in layers_with_rcnot constructs the RCNOT in the layers we specified in the array above
                # if (j != 1) or (i in layers_with_rcnot):
                if i in layers_with_rcnot:
                    circuit.cx(qbit, qreg_q[j+1])
    for j, qbit in enumerate(qreg_q):
        circuit.ry(weights[q_depth * n_qubits + j], qbit)
            
    circuit.measure(qreg_q, creg)
    return circuit
    

def run_circuit(circuit: QuantumCircuit) -> dict:
    """
    Runs a quantum circuit on a simulator and retrieves measurement counts.
    
    :param circuit: QuantumCircuit object.
    :returns: Dictionary of measurement counts.
    """
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=config.N_SHOTS, memory=True)
    counts = job.result().get_counts()
    return counts


def calculate_loss(y_true, y_pred):
    """
    Calculates the log loss between true and predicted labels.
    
    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :returns: Log loss value.
    """
    loss = log_loss(y_true, y_pred, labels=[0,1])
    return loss


@global_timer.timer
def train_baseline(X: list[float], y: list[Literal[0,1]], params: list[float], num_iter: int, n_qubits: int, q_depth: int) -> Tuple[list[float], list[float], list[float]]:
    """
    Train the baseline model with COBYLA optimization.
    
    :param X: Dataset samples.
    :param y: Dataset labels.
    :param params: Trainable weights.
    :param num_iter: Maximum number of epochs.
    :param n_qubits: Number of qubits.
    :param q_depth: Number of parametrized layers
    
    :returns: Losses, accuracy scores, and optimized weights.
    """
    iteration = 0
    all_losses = []
    all_accs = []
    all_predictions = []
    all_weights = []
    params_split_index = n_thetas = len(config.INITIAL_THETAS)
    # function to optimize
    # runs all data through our small network and computes the loss
    # returns the loss as the opitmization goal
    @global_timer.timer
    def method_to_optimize(params, samples, ys):
        nonlocal iteration
        print(f"Entering iteration {iteration}")
        thetas, interpret_weights = np.split(params, [params_split_index]) # split up trained weights
        iter_results = np.empty(len(X))
        iter_preds = np.empty(len(X))
        for i in range(len(samples)):
            circuit = create_circuit(n_qubits, q_depth, samples[i], thetas)
            counts = run_circuit(circuit)
            iter_results[i] = calculate_parity(counts)
            #iter_preds[i] = round(iter_results[i]) # rounded probability of belonging to class 1 = predicted label, only in case exp values are used
        all_predictions.append(iter_preds)
        all_weights.append(params)
        loss = calculate_loss(ys, iter_results)
        all_losses.append(loss)
        acc = accuracy_score(ys, iter_preds)
        all_accs.append(acc)
        print(f"Values in iteration {iteration}: Loss: {loss}, Accuracy: {acc}")
        iteration += 1
        # prediction as iter results
        return loss
    # callback function executed after every iteration of the minimize function        
    def iteration_callback(intermediate_params):
        print("Intermediate params: ", intermediate_params)
        return True
        
    # minimize gradient free
    constraints = [
        {'type': 'ineq', 'fun': lower_bound_constraint_with_split, 'args': (params_split_index, )},
        {'type': 'ineq', 'fun': upper_bound_constraint_with_split, 'args': (params_split_index, )}
    ]
    res = minimize(method_to_optimize, params, args=(X, y), options={'disp': True, 'maxiter': num_iter}, method=config.OPTIM_METHOD, callback=iteration_callback, constraints=constraints)
    save_losses_weights_predictions(f"debug_results_{config.OPTIM_METHOD}.csv", losses=all_losses, weights=all_weights, predictions=all_predictions)
    # return losses and accuracies for plotting
    # return last (optimized) weights for testing
    return all_losses, all_accs, all_weights[-1]


def load_dataset(dataset_str: str, n_samples: int) -> Tuple[list, list]:
    """
    Loads a dataset based on the provided dataset string.
    
    :param dataset_str: Dataset identifier.
    :param n_samples: Number of samples.
    :raises ValueError: in case no valid dataset_str provided
    :returns: Processed dataset.
    """
    if dataset_str.casefold() == "iris":
        return prepare_dataset_iris(config.N_QUBITS)
    elif dataset_str.casefold() == "moons":
        return prepare_dataset_moons(n_samples)
    else:
        raise ValueError("No valid dataset provided in config") 


def test_baseline(X: list[float], y: list[Literal[0,1]], params: list[float], n_qubits: int, q_depth: int, n_thetas: int = len(config.INITIAL_THETAS)) -> dict:
    """
    Tests the trained model.
    
    :param X: Test data samples.
    :param y: Test data labels.
    :param params: Trained model weights.
    :param n_qubits: Number of qubits.
    :param q_depth: Number of parametrized layers.
    :param n_thetas: Number of trainable circuit weights.
    
    :returns: Testing reprt.
    """
    test_results = np.empty(len(X))
    for i in range(len(X)):
        thetas, interpret_weights = np.split(params, [n_thetas]) # split in case we use trainable interpret weights for parity calculation
        circuit = create_circuit(n_qubits, q_depth, X[i], thetas)
        counts = run_circuit(circuit)
        predicted_label = calculate_parity(counts)
        test_results[i] = predicted_label
    # generate classification report
    dict_report = classification_report(y_true=y, y_pred=test_results, output_dict=True)
    return dict_report


def main() -> float:
    """
    Train and test the model and save all produced results.
    
    :returns: Testing accuracy.
    """
    # load the dataset
    X, y = load_dataset(config.DATASET_FUNCTION, config.SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y)
    weights_to_optimize = np.concatenate((config.INITIAL_THETAS, config.INITIAL_INTERPRET_WEIGHTS))
    losses, accs, weights = train_baseline(X_train, y_train, weights_to_optimize, config.NUM_ITER, config.N_QUBITS, config.Q_DEPTH)
    filename = f"qiskit_{config.DATASET_FUNCTION}_{config.OPTIM_METHOD}_{config.N_SHOTS}shots_{config.Q_DEPTH}depth_{config.SAMPLES}samples_{config.FEATURE_MAP}fmap_{config.N_QUBITS}qubits_{config.NUM_ITER}iters_seed{config.RANDOM_SEED}{config.FILENAME_ADDON}"
    plot_acc_and_loss("accs_loss_" + filename, accs, losses)
    # save weights
    save_weights_config_test_data_losses_accs(weights, X_test, y_test, losses, accs, filename)
    report = test_baseline(X_test, y_test, weights, config.N_QUBITS, config.Q_DEPTH)
    report["execution_avgs"] = global_timer.get_execution_averages()
    report["execution_times"] = global_timer.get_execution_times()
    save_classification_report(report, filename)
    # save circuit
    save_circuit(create_circuit, filename)
    return report['accuracy']
    
    
def test_multiple_runs_accs() -> list[float]:
    """
    Train and tests over multiple random seeds.
    
    :returns: Testing accuracy scores
    """
    seeds = [73,84,123,5,42,17,255,185,48,216]
    accs = []
    for i in seeds:
        config.RANDOM_SEED = i
        np.random.seed(config.RANDOM_SEED)
        acc = main()
        accs.append(acc)
    print(accs)    
    return accs 

if __name__ == "__main__":
    main()
    #test_multiple_runs_accs()
