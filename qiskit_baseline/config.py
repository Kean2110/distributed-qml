import math
import numpy as np
N_QUBITS = 2
NUM_ITER = 1000
RANDOM_SEED = 42
Q_DEPTH = 8
# either MOONS or IRIS
DATASET_FUNCTION = "MOONS"
OPTIM_METHOD = "cobyla"
N_SHOTS = 32
SAMPLES = 100
TEST_SIZE = 0.2
np.random.seed(RANDOM_SEED)
INITIAL_THETAS = np.random.uniform(0, 2*math.pi, (Q_DEPTH + 1) * N_QUBITS)
#INITIAL_INTERPRET_WEIGHTS = np.array([-1., 1., 1., -1.])
INITIAL_INTERPRET_WEIGHTS = np.array([]) # set array empty in case we do not want to train those weights
FEATURE_MAP = "RY"
FILENAME_ADDON = ""
LB_THETAS = 0.0
UB_THETAS = math.pi * 2
LB_INTERPRET = -np.inf
UB_INTERPRET = np.inf
