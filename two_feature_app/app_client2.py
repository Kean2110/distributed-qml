from netqasm.sdk.classical_communication.broadcast_channel import BroadcastChannelBySockets
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.logging.glob import get_netqasm_logger
from helper_functions import receive_teleported_qubit, send_value, receive_dict, teleport_qubit
from netqasm.sdk.toolbox import set_qubit_state

logger = get_netqasm_logger()

def main(app_config=None):
    socket_server = Socket("client2", "server", socket_id=1)
    socket_client1 = Socket("client2", "client1", socket_id=0)
    epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=0, remote_epr_socket_id=1)
    epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=1, remote_epr_socket_id=1)
    
    client2 = NetQASMConnection(
        app_name="client2",
        epr_sockets=[epr_socket_server, epr_socket_client1],
    )
    
    with client2:
        # receive number of iters
        params = receive_dict(socket_server)
        logger.info(f"Client2 received params dict: {params}")
        n_iters = params['n_iters']
        n_samples = params['n_samples']
        batch_size = params['batch_size']
        n_batches = params['n_batches']
        n_thetas = params['n_thetas']
        for _ in range(n_iters):
            for _ in range(n_samples):
                for _ in range(n_thetas + 1):
                    # receive feature from server
                    feature = float(socket_server.recv(block=True))
                    logger.info(f"Client 2 received feature: {feature}")
                    
                    # receive weight from server
                    theta = float(socket_server.recv(block=True))
                    logger.info(f"Client 2 received weight: {theta}")
                    
                    # receive teleported qubit from client1
                    qubit_recv = receive_teleported_qubit(epr_socket_client1, socket_client1, client2)
                    
                    teleport_qubit(epr_socket_client1, socket_client1, client2, feature, theta)
                    
                    q = Qubit(client2)
                    set_qubit_state(q, feature, theta)
                    qubit_recv.cnot(q)
                    
                    # measure
                    result = q.measure()
                    client2.flush()
                    
                    # send result to server
                    send_value(channel=socket_server, value=result)
            
        
if __name__ == "__main__":
    main()
