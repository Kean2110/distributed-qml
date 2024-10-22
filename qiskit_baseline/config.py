import math
import numpy as np
N_QUBITS = 2
NUM_ITER = 1000
RANDOM_SEED = 42
Q_DEPTH = 4
# either MOONS or IRIS
DATASET_FUNCTION = "IRIS"
OPTIM_METHOD = "cobyla"
N_SHOTS = 32
SAMPLES = 10000
TEST_SIZE = 0.2
np.random.seed(RANDOM_SEED)
INITIAL_THETAS = np.random.uniform(0, 2*math.pi, (Q_DEPTH + 1) * N_QUBITS)
#INITIAL_THETAS = np.random.rand((Q_DEPTH+1)*N_QUBITS)
FEATURE_MAP = "RY"
FILENAME_ADDON = "_constraints_params_features_scipy_1_10"