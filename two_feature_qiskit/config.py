import numpy as np
N_QUBITS = 2
NUM_ITER = 1000
RANDOM_SEED = 42
Q_DEPTH = 4
# either MOONS or IRIS
DATASET_FUNCTION = "MOONS"
OPTIM_METHOD = "cobyla"
N_SHOTS = 1024
SAMPLES = 1000
np.random.seed(RANDOM_SEED)
INITIAL_THETAS = np.random.rand((Q_DEPTH+1)*N_QUBITS)