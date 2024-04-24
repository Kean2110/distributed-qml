from netqasm.sdk import Qubit
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk.toolbox import set_qubit_state
from utils.logger import logger


def receive_teleported_qubit(epr_socket, classical_socket, netqasm_connection):
    with netqasm_connection:
        epr = epr_socket.recv_keep()[0]
        netqasm_connection.flush()

        m1, m2 = classical_socket.recv_structured().payload
        print(f"`receiver` got corrections: {m1}, {m2}")
        if m2 == 1:
            print("`receiver` will perform X correction")
            epr.X()
        if m1 == 1:
            print("`receiver` will perform Z correction")
            epr.Z()
        netqasm_connection.flush()
        return epr


def teleport_qubit(epr_socket, classical_socket, netqasm_connection, feature, theta):
    with netqasm_connection:
        q = Qubit(netqasm_connection)
        set_qubit_state(q, feature, theta)

        epr = epr_socket.create_keep()[0]

        # Teleport
        q.cnot(epr)
        q.H()
        m1 = q.measure()
        m2 = epr.measure()

    # send corrections
    m1, m2 = int(m1), int(m2)

    classical_socket.send_structured(StructuredMessage("Corrections", (m1,m2)))


def remote_cnot_control(classical_socket: Socket, control_qubit: Qubit, epr_qubit: Qubit):
    conn = epr_qubit.connection
    # CNOT between ctrl and epr
    control_qubit.cnot(epr_qubit)
    
    # measure epr
    epr_ctrl_meas = epr_qubit.measure()
    
    conn.flush()
    logger.debug(f"measured {epr_ctrl_meas}")

    classical_socket.send(str(epr_ctrl_meas))

    # wait for target's measurement outcome to undo potential entanglement
    # between his EPR half and the control qubit
    target_meas = classical_socket.recv(block=True)
    conn = control_qubit.connection
    if target_meas == "1":
        control_qubit.Z()
        conn.flush()


def remote_cnot_target(classical_socket: Socket, target_qubit: Qubit, epr_qubit: Qubit):

    # receive measurement result from EPR pair from controller
    epr_meas = classical_socket.recv(block=True)

    # apply X gate if control epr qubit is 1
    if epr_meas == "1":
        epr_qubit.X()

    # apply CNOT between EPR Qubit and target qubit
    epr_qubit.cnot(target_qubit)

    # apply H gate to epr target qubit and measure it and send it to controller
    epr_qubit.H()

    # undo any potential entanglement between `epr` and controller's control qubit
    epr_target_meas = epr_qubit.measure()
    conn = epr_qubit.connection
    conn.flush()

    # Controller will do a controlled-Z based on the outcome to undo the entanglement
    classical_socket.send(str(epr_target_meas))