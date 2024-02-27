from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.sdk.toolbox.sim_states import get_fidelity, qubit_from, to_dm
from teleport_qubit import remote_cnot_control

def main(app_config=None):
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
        q0 = Qubit(node1)
        q1 = Qubit(node1)
        q0.X()
        q0.cnot(q1)
        
        remote_cnot_control(epr_socket, socket, node1, q1)

        node1.flush()
        q0_state = get_qubit_state(q0)
        q1_state = get_qubit_state(q1)
        print(
            f"Qubit state q0: {q0_state.tolist()}, Qubit state q1: {q1_state.tolist()}"
        )
        return {"q0_state": q0_state.tolist(), "q1_state": q1_state.tolist()}


if __name__ == "__main__":
    main()
