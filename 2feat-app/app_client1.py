from netqasm.sdk.classical_communication.broadcast_channel import BroadcastChannelBySockets
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.logging.glob import get_netqasm_logger
from helper_functions import send_value, receive_dict, teleport_qubit, receive_teleported_qubit
from netqasm.sdk.toolbox import set_qubit_state

logger = get_netqasm_logger()
    
def main(app_config=None):
    socket_server = Socket("client1", "server", socket_id=0)
    socket_client2 = Socket("client1", "client2", socket_id=1)
    epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=0, remote_epr_socket_id=0)
    epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=1, remote_epr_socket_id=1)
    
    client1 = NetQASMConnection(
        app_name="client1",
        epr_sockets=[epr_socket_server, epr_socket_client2],
    )
    
    with client1:
        # receive params
        params = receive_dict(socket_server)
        logger.info(f"Client1 received params dict: {params}")
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
                    logger.info(f"Client 1 received feature: {feature}")
                    
                    # receive weight from server
                    theta = float(socket_server.recv(block=True))
                    logger.info(f"Client 1 received weight: {theta}")
                    
                    teleport_qubit(epr_socket_client2, socket_client2, client1, feature, theta)
                    
                    qubit_recv = receive_teleported_qubit(epr_socket_client2, socket_client2, client1)
                    
                    q = Qubit(client1)
                    set_qubit_state(q, feature, theta)

                    qubit_recv.cnot(q)
                    
                    # measure
                    result = q.measure()
                    client1.flush()
                    
                    # send result to server
                    send_value(channel=socket_server, value=result)        
            

if __name__ == "__main__":
    main()
