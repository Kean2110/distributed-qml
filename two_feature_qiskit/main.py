import math
import os
from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, BasicAer, execute
from sklearn.metrics import log_loss, accuracy_score
from helper_functions import check_parity
from scipy.optimize import minimize
import numpy as np
import random
import config
import helper_functions
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


def create_RX_feature_map(n_qubits, features):
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    for i, qubit in enumerate(qreg):
        circuit.rx(features[i], qubit)
    return circuit
            

def create_circuit(n_qubits, q_depth, features, weights):
    assert len(weights) == n_qubits * q_depth, "Number of weights doesn't match n_qubits * q_depth"
    # Two qubits
    qreg_q = QuantumRegister(n_qubits, 'q')
    creg = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qreg_q, creg)
    
    feature_map = create_ZZ_feature_map(n_qubits, features)
    #feature_map = create_RX_feature_map(n_qubits, features)
    circuit = circuit.compose(feature_map)
    # apply weights // entanglement layer
    for i in range(q_depth):
        for j, qbit in enumerate(qreg_q):
            circuit.ry(weights[j * q_depth + i], qbit)
        for j, qbit in enumerate(qreg_q):
            if j < len(qreg_q) - 1:
                circuit.cx(qbit, qreg_q[j+1])
            
    circuit.measure(qreg_q, creg)
    return circuit
    

def run_circuit(circuit: QuantumCircuit) -> list[int]:
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=config.N_SHOTS, memory=True)
    counts = job.result().get_counts()
    string_qubit_values = max(counts, key=lambda key: counts[key])
    qubit_results = []
    for i in range(len(string_qubit_values)):
        qubit_results.append(int(string_qubit_values[i]))
    return qubit_results


def calculate_loss(y_true, y_pred):
    loss = log_loss(y_true, y_pred, labels=[0,1])
    return loss


def run_gradient_free(X, y, thetas, num_iter, n_qubits, q_depth):
    iteration = 0
    all_losses = []
    all_accs = []
    # function to optimize
    # runs all data through our small network and computes the loss
    # returns the loss as the opitmization goal
    def method_to_optimize(params, samples, ys):
        nonlocal iteration
        print(f"Entering iteration {iteration}")
        iter_results = np.empty(len(X))
        for i in range(len(samples)):
            circuit = create_circuit(n_qubits, q_depth, samples[i], params)
            qubit_results = run_circuit(circuit)
            predicted_label = helper_functions.check_parity(qubit_results)
            iter_results[i] = predicted_label
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
        return True
        
    # minimize gradient free
    res = minimize(method_to_optimize, thetas, args=(X, y), options={'disp': True, 'maxiter': num_iter}, method=config.OPTIM_METHOD, callback=iteration_callback)
    return all_losses, all_accs


def load_dataset(dataset_str):
    if dataset_str.casefold() == "iris":
        return helper_functions.prepare_dataset_iris()
    elif dataset_str.casefold() == "moons":
        return helper_functions.prepare_dataset_moons()
    else:
        raise ValueError("No valid dataset provided in config") 
    

def main():
    # load the dataset
    X, y = load_dataset(config.DATASET_FUNCTION)
    print(np.shape(X), np.shape(y))
    losses, accs = run_gradient_free(X, y, config.INITIAL_THETAS, config.NUM_ITER, config.N_QUBITS, config.Q_DEPTH)
    filename = "qiskit_optimizer_" + config.DATASET_FUNCTION+ "_" + config.OPTIM_METHOD + "_" + str(config.N_SHOTS) + "_" + str(config.Q_DEPTH)
    helper_functions.plot_acc_and_loss("accs_loss_" + filename, accs, losses)

if __name__ == "__main__":
    '''
    circuit = create_circuit(2, config.Q_DEPTH, [0,0], np.random.rand(2*config.Q_DEPTH))
    print(circuit.draw())
    qbit_results = run_circuit(circuit)
    print(qbit_results)
    '''
    main()
