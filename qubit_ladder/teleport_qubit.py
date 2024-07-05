from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import Socket, NetQASMConnection
from netqasm.sdk.classical_communication.message import StructuredMessage


def teleport_qubit(qubit, conn, classical_socket, epr_socket):
    with conn:
        # Create EPR pairs
        epr = epr_socket.create_keep()[0]

        # Teleport
        qubit.cnot(epr)
        qubit.H()
        m1 = qubit.measure()
        m2 = epr.measure()

    # send correction information
    print(
        f"`sender` measured the following teleportation corrections: m1 = {m1}, m2 = {m2}"
    )
    print("`sender` will send the corrections to `receiver`")

    classical_socket.send_structured(StructuredMessage("Corrections", (m1, m2)))


def receive_teleported_qubit(conn, classical_socket, epr_socket):
    with conn:
        epr = epr_socket.recv_keep()[0]
        conn.flush()

        # Get the corrections
        m1, m2 = classical_socket.recv_structured().payload
        print(f"`receiver` got corrections: {m1}, {m2}")
        if m2 == 1:
            print("`receiver` will perform X correction")
            epr.X()
        if m1 == 1:
            print("`receiver` will perform Z correction")
            epr.Z()

        conn.flush()
    return epr


def remote_cnot_control(epr_socket: EPRSocket, classical_socket: Socket, netqasm_conn: NetQASMConnection, control_qubit: Qubit):
    assert classical_socket.recv() == "ACK"
    # create EPR pair
    epr_ctrl = epr_socket.create_keep()[0]
    
    # CNOT between ctrl and epr
    control_qubit.cnot(epr_ctrl)
    
    # measure epr
    epr_ctrl_meas = epr_ctrl.measure()
    netqasm_conn.flush()
    
    classical_socket.send(str(epr_ctrl_meas))
    
    # wait for target's measurement outcome to undo potential entanglement
    # between his EPR half and the control qubit
    target_meas = classical_socket.recv()
    if target_meas == "1":
        control_qubit.Z()
    netqasm_conn.flush()
        
        
def remote_cnot_target(epr_socket: EPRSocket, classical_socket: Socket, netqasm_conn: NetQASMConnection, target_qubit: Qubit):
    classical_socket.send("ACK")
    
    # receive EPR qubit
    epr_target = epr_socket.recv_keep()[0]
    netqasm_conn.flush()

    # receive measurement result from EPR pair from controller
    epr_meas = classical_socket.recv()

    # apply X gate if control epr qubit is 1
    if epr_meas == "1":
        epr_target.X()

    # apply CNOT between EPR Qubit and target qubit
    epr_target.cnot(target_qubit)

    # apply H gate to epr target qubit and measure it and send it to controller
    epr_target.H()

    # undo any potential entanglement between `epr` and controller's control qubit
    epr_target_meas = epr_target.measure()
    netqasm_conn.flush()

    # Controller will do a controlled-Z based on the outcome to undo the entanglement
    classical_socket.send(str(epr_target_meas))
