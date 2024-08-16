import numpy as np
N_QUBITS = 4
NUM_ITER = 100
RANDOM_SEED = 42
Q_DEPTH = 4
# either MOONS or IRIS
DATASET_FUNCTION = "IRIS"
OPTIM_METHOD = "cobyla"
N_SHOTS = 32
SAMPLES = 1000
TEST_SIZE = 0.2
np.random.seed(RANDOM_SEED)
INITIAL_THETAS = np.random.rand((Q_DEPTH+1)*N_QUBITS)
FEATURE_MAP = "RY"
FILENAME_ADDON = "_first_and_last_rcnot"