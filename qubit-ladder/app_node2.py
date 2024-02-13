from netqasm.logging.output import get_new_app_logger
from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.sdk.toolbox import set_qubit_state


def main(app_config=None, phi=0.0, theta=0.0):
    log_config = app_config.log_config
    # Create a socket to send classical information
    socket = Socket("node2", "node1", log_config=log_config)

    # Create a EPR socket for entanglement generation
    epr_socket = EPRSocket("node1")

    # Initialize the connection to the backend
    node2 = NetQASMConnection(
        app_name=app_config.app_name, log_config=log_config, epr_sockets=[epr_socket]
    )
    with node2:
        # Create two qubits
        q2 = Qubit(node2)
        q3 = Qubit(node2)
        
        # receive EPR pair
        epr_1 = epr_socket.recv_keep()[0]
        
        # receive measurement result from EPR pair from node1
        epr_0_meas = socket.recv()
        
        # apply X gate if epr_0 is 1
        if epr_0_meas == "1":
            epr_1.X()
            
        # apply CNOT between EPR Qubit and q2
        epr_1.cnot(q2)
        
        # CNOT between q2 and q3
        q2.cnot(q3)
        
        # apply H gate to epr_1 and measure it and send it to node1
        epr_1.H()
        # undo any potential entanglement between `epr` and Controller's control qubit
        epr_1_meas = epr_1.measure()
        node2.flush()

        # Node1 will do a controlled-Z based on the outcome to undo the entanglement
        socket.send(str(epr_1_meas))
            
        q2_state = get_qubit_state(q2)
        q3_state = get_qubit_state(q3)
        print(f"Qubit state q0: {q2_state.tolist()}, Qubit state q1: {q3_state.tolist()}")
        return{
            "q2_state": q2_state.tolist(),
            "q3_state": q3_state.tolist()
        }
        
        
        
        



if __name__ == "__main__":
    main()
