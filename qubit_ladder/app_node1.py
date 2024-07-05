import os
#os.environ["NETQASM_SIMULATOR"] = "netsquid_single_thread"
from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.sdk.toolbox.sim_states import get_fidelity, qubit_from, to_dm
from teleport_qubit import remote_cnot_control
from netqasm.logging.glob import get_netqasm_logger


def main(app_config=None):
    logger = get_netqasm_logger()
    log_config = app_config.log_config
    # Create a socket to communicate classical information
    socket = Socket("node1", "node2", log_config=log_config)

    # Create a EPR socket for entanglement generation
    epr_socket = EPRSocket("node2")

    # Initialize the connection
    node1 = NetQASMConnection(
        app_name="node1", log_config=log_config, epr_sockets=[epr_socket]
    )
    with node1:
        for i in range(500):
            logger.info(f"Enter iter {i}")
            q0 = Qubit(node1)
            q1 = Qubit(node1)
            q0.X()
            q0.cnot(q1)
            node1.flush()
            remote_cnot_control(epr_socket, socket, node1, q1)
            
            q0_state = get_qubit_state(q0)
            q1_state = get_qubit_state(q1)
            print(
                f"Qubit state q0: {q0_state.tolist()}, Qubit state q1: {q1_state.tolist()}"
            )
            q0_res = q0.measure()
            q1_res = q1.measure()
            node1.flush()
            print(f"Measurement results; Q0 : {q0_res}, Q1: {q1_res}")


if __name__ == "__main__":
    main()
