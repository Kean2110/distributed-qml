from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.sdk.toolbox.sim_states import get_fidelity, qubit_from, to_dm


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
        # create EPR pair
        epr_0 = epr_socket.create_keep()[0]
        node1.flush()

        # create two qubits
        q0 = Qubit(node1)
        q1 = Qubit(node1)

        q0.X()
        # q1.X()

        # CNOT between q0 and q1
        q0.cnot(q1)

        # CNOT between q1 and epr_0
        q1.cnot(epr_0)

        # measure epr_0
        epr_meas = epr_0.measure()

        # flush to perform qubit operations
        node1.flush()

        # send classical information to node2
        socket.send(str(epr_meas))

        # wait for node2's measurement outcome to undo potential entanglement
        # between his EPR half and the node1's control qubits
        target_meas = socket.recv()
        if target_meas == "1":
            q1.Z()

        node1.flush()

        q0_state = get_qubit_state(q0)
        q1_state = get_qubit_state(q1)
        print(
            f"Qubit state q0: {q0_state.tolist()}, Qubit state q1: {q1_state.tolist()}"
        )
        return {"q0_state": q0_state.tolist(), "q1_state": q1_state.tolist()}


if __name__ == "__main__":
    main()
