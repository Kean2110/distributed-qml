import os
#os.environ["NETQASM_SIMULATOR"] = "netsquid_single_thread"
from netqasm.logging.output import get_new_app_logger
from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.toolbox import set_qubit_state
from teleport_qubit import remote_cnot_target

def main(app_config=None, phi=0.0, theta=0.0):
    logger = get_netqasm_logger()
    log_config = app_config.log_config
    # Create a socket to send classical information
    socket = Socket("node2", "node1", log_config=log_config)

    # Create a EPR socket for entanglement generation
    epr_socket = EPRSocket("node1")

    # Initialize the connection to the backend
    node2 = NetQASMConnection(
        app_name="node2", log_config=log_config, epr_sockets=[epr_socket]
    )
    with node2:
        for i in range(500):
            logger.info(f"Enter iter {i}")
            # target qubit
            q2 = Qubit(node2)
            q3 = Qubit(node2)
            q2.X()
            node2.flush()
            remote_cnot_target(epr_socket, socket, node2, q2)
            # build ladder
            q3.X()
            q2.cnot(q3)
            node2.flush()
            
            q2_state = get_qubit_state(q2)
            q3_state = get_qubit_state(q3)
            print(
                f"Qubit state q2: {q2_state.tolist()}, Qubit state q3: {q3_state.tolist()}"
            )
            
            q2_res = q2.measure()
            q3_res = q3.measure()
            node2.flush()
            print(f"Measurement results; Q2 : {q2_res}, Q3: {q3_res}")


if __name__ == "__main__":
    main()
