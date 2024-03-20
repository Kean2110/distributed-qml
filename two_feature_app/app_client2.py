from netqasm.sdk.classical_communication.broadcast_channel import BroadcastChannelBySockets
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.logging.glob import get_netqasm_logger
from helper_functions import receive_teleported_qubit, send_value, receive_dict, teleport_qubit, remote_cnot_target
from netqasm.sdk.toolbox import set_qubit_state

logger = get_netqasm_logger()

def main(app_config=None):
    try:
        run_client2()
    except Exception as e:
        print("An error occured in client 2: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()

def run_client2():
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
    q_depth  = params['q_depth']

    with client2:
        while True:
            instruction = socket_server.recv(block=True)
            if instruction == "EXIT":
                break
            elif instruction == "RUN CIRCUIT":
                run_circuit_locally(client2, socket_client1, socket_server, epr_socket_client1, q_depth)
            else:
                raise ValueError("Unregistered instruction received")                        


def run_circuit_locally(conn: NetQASMConnection, socket_client1: Socket, socket_server: Socket, epr_socket_client1: EPRSocket, q_depth: int):
    feature = float(socket_server.recv(block=True))
    logger.info(f"Client 2 received feature: {feature}")
    
    # generate epr pairs
    eprs = epr_socket_client1.recv_keep(number=q_depth)
    logger.info(f"Client 2 received {q_depth} epr pairs")
        
    # set up local qubit
    q2 = Qubit(conn)
    q2.rot_X(angle=feature)
    
    # execute hidden layer
    for i in range(q_depth):
        logger.info(f"Client 2 entered layer: {i}")
        
        # receive weight from server
        theta = float(socket_server.recv(block=True))
        logger.info(f"Client 2 received weight: {theta}")
        
        # apply theta rotation
        q2.rot_Y(angle=theta)
    
        # remote cnot with qubit of client1
        remote_cnot_target(socket_client1, conn, q2, eprs[i])
    
    # measure
    result = q2.measure()
    conn.flush()
        
    # send result to server
    send_value(channel=socket_server, value=result)

    

      
if __name__ == "__main__":
    main()
