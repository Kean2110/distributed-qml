import math
import os

# Instructions
EXIT_INSTRUCTION = "EXIT"
RUN_INSTRUCTION = "RUN ITERATION"
TEST_INSTRUCTION = "TEST"


# Message Headers
OWN_FEATURES = "OWN FEATURES"
OTHER_FEATURES = "OTHER FEATURES"
TEST_FEATURES = "TEST FEATURES"
PARAMS = "PARAMS"
THETAS = "THETAS"
RESULTS = "RESULTS"


# SOCKET IDs
# TYPE_NODE1_NODE2_SOCKET@NODE
EPR_SERVER_C1_SERVER = 0
EPR_SERVER_C1_C1 = 0
EPR_SERVER_C2_SERVER = 1
EPR_SERVER_C2_C2 = 0
EPR_C1_C2_C1 = 1
EPR_C1_C2_C2 = 1
SOCKET_SERVER_C1 = 0
SOCKET_SERVER_C2 = 1
SOCKET_C1_C2 = 2

APP_BASE_PATH = os.getcwd()

# MAXIMUM Values defined
# MAX DEPTH OF 4 RIGHT NOW BECAUSE ONLY 5 QUBITS ARE SUPPORTED!
MAX_VALUES = {
    "q_depth": 12,
    "eprs": 4,
    "qubits_per_client": 8
}

MIN_VALUES = {
    "q_depth": 1,
    "n_samples": 10,
    "n_shots": 1,
    "n_qubits": 2
}

# EPSILON added onto the lower bounds to prevent wrong angle calculations
BOUND_EPSILON = 1e-4

# BOUNDS constraints for the params
LOWER_BOUND_PARAMS = 0 + BOUND_EPSILON
UPPER_BOUND_PARAMS = 2 * math.pi

LOWER_BOUND_INPUTS = 0 + BOUND_EPSILON
UPPER_BOUND_INPUTS = 1

SINGLE_THREAD_SIMULATOR = "netsquid_single_thread"
MULTI_THREAD_SIMULATOR = "netsquid"

NODE_NAMES = ["server", "client1", "client2"]
DEFAULT_HW = QuantumHardware.Generic