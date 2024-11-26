import math
import os
from typing import Literal
from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, BasicAer, execute
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
from helper_functions import check_parity, lower_bound_constraint, prepare_dataset_iris, prepare_dataset_moons, plot_acc_and_loss, plot_accuracy, plot_losses, save_circuit, save_losses_weights_predictions, save_classification_report, save_weights_config_test_data, upper_bound_constraint
from scipy.optimize import minimize, Bounds
import numpy as np
import random
import config
import time



def create_ZZ_feature_map(n_qubits, features):
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


def create_rot_feature_map(n_qubits, features):
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    for i, qubit in enumerate(qreg):
        circuit.ry(features[i], qubit)
    return circuit
            

def create_circuit(n_qubits, q_depth, features, weights):
    assert len(weights) == n_qubits * (q_depth+1), "Number of weights doesn't match n_qubits * (q_depth+1)"
    # Specify register
    qreg_q = QuantumRegister(n_qubits, 'q')
    creg = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qreg_q, creg)
    
    #feature_map = create_ZZ_feature_map(n_qubits, features)
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
    

def run_circuit(circuit: QuantumCircuit) -> list[int]:
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=config.N_SHOTS, memory=True)
    counts = job.result().get_counts()
    return counts


def calculate_parity(counts: dict) -> list:
    zeros = 0
    ones = 0
    for measure, count in counts.items():
        if measure.count('1') % 2 == 0:
            zeros += count
        else:
            ones += count
    output_probs = [zeros / config.N_SHOTS, ones / config.N_SHOTS]
    return output_probs.index(max(output_probs))


def calculate_loss(y_true, y_pred):
    #loss = brier_score_loss(y_true, y_pred)
    loss = log_loss(y_true, y_pred, labels=[0,1])
    return loss


def run_gradient_free(X, y, thetas, num_iter, n_qubits, q_depth):
    iteration = 0
    all_losses = []
    all_accs = []
    all_predictions = []
    all_weights = []
    # function to optimize
    # runs all data through our small network and computes the loss
    # returns the loss as the opitmization goal
    def method_to_optimize(params, samples, ys):
        nonlocal iteration
        print(f"Entering iteration {iteration}")
        iter_results = np.empty(len(X))
        for i in range(len(samples)):
            circuit = create_circuit(n_qubits, q_depth, samples[i], params)
            counts = run_circuit(circuit)
            predicted_label = calculate_parity(counts)
            iter_results[i] = predicted_label
        all_predictions.append(iter_results)
        all_weights.append(params)
        loss = calculate_loss(ys, iter_results)
        all_losses.append(loss)
        acc = accuracy_score(ys, iter_results)
        all_accs.append(acc)
        print(f"Values in iteration {iteration}: Loss: {loss}, Accuracy: {acc}")
        iteration += 1
        # prediction as iter results
        return loss
    # callback function executed after every iteration of the minimize function        
    def iteration_callback(intermediate_params):
        print("Intermediate thetas: ", intermediate_params)
        print(f"Max theta: {max(intermediate_params)}, min theta: {min(intermediate_params)}")
        return True
        
    # minimize gradient free
    constraints = [
        {'type': 'ineq', 'fun': lower_bound_constraint},
        {'type': 'ineq', 'fun': upper_bound_constraint}
    ]
    bounds = Bounds(0, 2*math.pi)
    res = minimize(method_to_optimize, thetas, args=(X, y), options={'disp': True, 'maxiter': num_iter}, method=config.OPTIM_METHOD, callback=iteration_callback, constraints=constraints)
    save_losses_weights_predictions(f"debug_results_{config.OPTIM_METHOD}.csv", losses=all_losses, weights=all_weights, predictions=all_predictions)
    # return losses and accuracies for plotting
    # return last (optimized) weights for testing
    return all_losses, all_accs, all_weights[-1]


def load_dataset(dataset_str, n_samples):
    if dataset_str.casefold() == "iris":
        return prepare_dataset_iris(config.N_QUBITS)
    elif dataset_str.casefold() == "moons":
        return prepare_dataset_moons(n_samples)
    else:
        raise ValueError("No valid dataset provided in config") 


def test(X, y, thetas, n_qubits, q_depth):
    test_results = np.empty(len(X))
    for i in range(len(X)):
        circuit = create_circuit(n_qubits, q_depth, X[i], thetas)
        counts = run_circuit(circuit)
        predicted_label = calculate_parity(counts)
        test_results[i] = predicted_label
    # generate classification report
    dict_report = classification_report(y_true=y, y_pred=test_results, output_dict=True)
    return dict_report


def main():
    # load the dataset
    X, y = load_dataset(config.DATASET_FUNCTION, config.SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y)
    losses, accs, weights = run_gradient_free(X_train, y_train, config.INITIAL_THETAS, config.NUM_ITER, config.N_QUBITS, config.Q_DEPTH)
    filename = f"qiskit_{config.DATASET_FUNCTION}_{config.OPTIM_METHOD}_{config.N_SHOTS}shots_{config.Q_DEPTH}depth_{config.SAMPLES}samples_{config.FEATURE_MAP}fmap_{config.N_QUBITS}qubits_{config.NUM_ITER}iters{config.FILENAME_ADDON}"
    plot_acc_and_loss("accs_loss_" + filename, accs, losses)
    # save weights
    save_weights_config_test_data(weights, X_test, y_test, filename)
    report = test(X_test, y_test, weights, config.N_QUBITS, config.Q_DEPTH)
    save_classification_report(report, filename)
    # save circuit
    save_circuit(create_circuit, filename)
    

if __name__ == "__main__":
    main()
    #print(create_circuit(config.N_QUBITS, config.Q_DEPTH, np.ones(config.N_QUBITS), np.zeros(config.N_QUBITS * (config.Q_DEPTH + 1))))
