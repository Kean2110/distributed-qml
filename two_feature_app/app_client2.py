from netqasm.sdk.classical_communication.broadcast_channel import BroadcastChannelBySockets
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.logging.glob import get_netqasm_logger
from helper_functions import receive_teleported_qubit, send_value, receive_dict, teleport_qubit, remote_cnot_target
from netqasm.sdk.toolbox import set_qubit_state

logger = get_netqasm_logger()

def main(app_config=None):
    socket_server = Socket("client2", "server", socket_id=1)
    socket_client1 = Socket("client2", "client1", socket_id=2)
    epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=0, remote_epr_socket_id=1)
    epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=1, remote_epr_socket_id=1)
    
    client2 = NetQASMConnection(
        app_name="client2",
        epr_sockets=[epr_socket_server, epr_socket_client1],
    )
    
    # receive number of iters
    params = receive_dict(socket_server)
    logger.info(f"Client2 received params dict: {params}")
    n_iters = params['n_iters']
    n_samples = params['n_samples']
    batch_size = params['batch_size']
    n_batches = params['n_batches']
    n_thetas = params['n_thetas']
    
    circuit_runner_count = 0
    with client2:
        while True:
            instruction = socket_server.recv(block=True)
            if instruction == "EXIT":
                break
            elif instruction == "RUN CIRCUIT":
                circuit_runner_count += 1
                print(f"Client2's circuit is running for the {circuit_runner_count} time")
                run_circuit_locally(client2, socket_client1, socket_server, epr_socket_client1)
            else:
                raise ValueError("Unregistered instruction received")                        

def run_circuit_locally(conn: NetQASMConnection, socket_client1: Socket, socket_server: Socket, epr_socket_client1: EPRSocket):
    feature = float(socket_server.recv(block=True))
    logger.info(f"Client 2 received feature: {feature}")
        
    # receive weight from server
    theta = float(socket_server.recv(block=True))
    logger.info(f"Client 2 received weight: {theta}")
    
    # receive EPR pair qubit
    epr2 = epr_socket_client1.recv_keep()[0]
    
    # set up local qubit
    q2 = Qubit(conn)
    set_qubit_state(q2, feature, theta)
    
    # remote cnot with qubit of client1
    remote_cnot_target(socket_client1, conn, q2, epr2)
    
    # measure
    result = q2.measure()
    conn.flush()
        
    # send result to server
    send_value(channel=socket_server, value=result)

    

      
if __name__ == "__main__":
    main()
