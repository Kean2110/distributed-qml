from netqasm.sdk.classical_communication.broadcast_channel import BroadcastChannelBySockets
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.logging.glob import get_netqasm_logger
from helper_functions import send_value, receive_dict, remote_cnot_control
from netqasm.sdk.toolbox import set_qubit_state

logger = get_netqasm_logger()
    
def main(app_config=None):
    socket_server = Socket("client1", "server", socket_id=0)
    socket_client2 = Socket("client1", "client2", socket_id=2)
    epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=0, remote_epr_socket_id=0)
    epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=1, remote_epr_socket_id=1)
    
    client1 = NetQASMConnection(
        app_name="client1",
        epr_sockets=[epr_socket_server, epr_socket_client2],
    )
    
    
    # receive params
    params = receive_dict(socket_server)
    logger.info(f"Client1 received params dict: {params}")
    circuit_runner_count = 0
    with client1: 
        while True:
            # receive instruction from server
            instruction = socket_server.recv(block=True)
            if instruction == "EXIT":
                break
            elif instruction == "RUN CIRCUIT":
                circuit_runner_count += 1
                print(f"Client1's circuit is running for the {circuit_runner_count} time")
                run_circuit_locally(client1, socket_client2, socket_server, epr_socket_client2)
            else:
                raise ValueError("Unregistered instruction received")  
                          
                

def run_circuit_locally(conn: NetQASMConnection, socket_client2: Socket, socket_server: Socket, epr_socket_client2: EPRSocket):
    # receive feature from server
    feature = float(socket_server.recv(block=True))
    logger.info(f"Client 1 received feature: {feature}")

    # receive weight from server
    theta = float(socket_server.recv(block=True))
    logger.info(f"Client 1 received weight: {theta}")

    # create EPR pair
    epr1 = epr_socket_client2.create_keep()[0]

    # initalize local qubit
    q1 = Qubit(conn)
    set_qubit_state(q1, feature, theta)

    # remote CNOT with qubit of client2
    remote_cnot_control(socket_client2, conn, q1, epr1)

    # measure
    result = q1.measure()
    conn.flush()
        
    # send result to server
    send_value(channel=socket_server, value=result)  

if __name__ == "__main__":
    main()
