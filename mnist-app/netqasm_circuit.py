import numpy as np

from netqasm.logging.glob import get_netqasm_logger, set_log_level
from netqasm.sdk.external import NetQASMConnection, simulate_application
from netqasm.sdk import Qubit, EPRSocket
from netqasm.runtime.application import default_app_instance
from preprocessing import preprocessing_amplitude_encoding

logger = get_netqasm_logger()

def encode_data_in_circuit(image: np.ndarray):
    flattened_image = np.ndarray.flatten(image)
    print(flattened_image.shape)
    with NetQASMConnection("encoding") as encoding:
        outcomes = encoding.new_array(len(flattened_image))
        values = encoding.new_array(len(flattened_image), init_values=flattened_image)
        with encoding.loop(len(flattened_image)) as i:
            q = Qubit(encoding)
            q.rot_X(values[i])
        encoding.flush()
    return circuit
    


def run_circuit():
    logger.debug("RUNNING CIRCUIT")
    logger.debug("STARTING PREPROCESSING")
    (x_train, y_train), (x_test, y_test) = preprocessing_amplitude_encoding()
    circ = encode_data_in_circuit(x_train[0])
    print(circ)


def create_app():
    app_instance = default_app_instance(
        [
            ("encoding", run_circuit)
        ]
    )
    
    simulate_application(
        app_instance,
        use_app_config=False,
        post_function=None,
        enable_logging=False
    )
    
if __name__ == "__main__":
    #set_log_level("DEBUG")
    create_app()