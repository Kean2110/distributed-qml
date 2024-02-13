from netqasm.sdk import EPRSocket, Qubit
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
    print(f"`sender` measured the following teleportation corrections: m1 = {m1}, m2 = {m2}")
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
    