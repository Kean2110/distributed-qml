from helper_functions import prepare_dataset_moons, prepare_dataset_iris 
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, NELDER_MEAD, POWELL
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import VQC

def create_rot_feature_map(n_qubits, features):
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    for i, qubit in enumerate(qreg):
        circuit.rx(features[i], qubit)
    return circuit

q_depth = 4
n_iters = 100

X, y = prepare_dataset_moons(100)

n_features = X.shape[1]
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1)

ansatz = RealAmplitudes(num_qubits=n_features, reps=q_depth)

optimizer = POWELL(maxiter=n_iters)
sampler = Sampler()

from matplotlib import pyplot as plt

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    print(weights)
    objective_func_vals.append(obj_func_eval)


def plot_results(values):
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(values)), values)
    plt.show()
     

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

vqc.fit(X,y)

plot_results(objective_func_vals)