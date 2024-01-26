from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from helper_functions import send_value, check_parity, split_data_into_batches, send_dict
import numpy as np


logger = get_netqasm_logger()

def prepare_dataset():
    """
    Loads the Iris dataset from scikit-learn, filters out the first two features from each sample
    and transforms it into a binary classification problem by ommitting one class.
    Then the features are normalized fit in the range [0,1]
    """
    iris = datasets.load_iris()
    # use only first two features of the iris dataset
    X = iris.data[:,:2]
    # filter out only zero and one classes
    filter_mask = np.isin(iris.target, [0,1])
    X_filtered = X[filter_mask]
    y_filtered = iris.target[filter_mask]
    # min max scale features to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled, y_filtered


def main(app_config=None, num_iter=1, theta_0=0, theta_1=0, batch_size=1):
    # setup classical socket connections
    socket_client1 = Socket("server", "client1", socket_id=0)
    socket_client2 = Socket("server", "client2", socket_id=1)
    # setup EPR connections
    epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=0, remote_epr_socket_id=0)
    epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=1, remote_epr_socket_id=0)
    server = NetQASMConnection(
        app_name="server",
        epr_sockets=[epr_socket_client1, epr_socket_client2],
    )
    X, y = prepare_dataset()

    batches = split_data_into_batches(X, batch_size)
    
    with server:
        
        # send parameters to the clients
        params = {'n_iters': num_iter, 'n_samples': len(X), 'batch_size': batch_size, 'n_batches': len(batches)}
        send_dict(socket_client1, params)
        send_dict(socket_client2, params)
        
        all_results = np.empty((num_iter, len(X)))
        for i in range(num_iter):
            print(f"ENTERED ITERATION {i+1}")
            for j, sample in enumerate(X):
                # Send first feature to client 1
                send_value(socket_client1, sample[0])
                # Send second feature to client 2
                send_value(socket_client2, sample[1])
                
                # Send theta0 to first client
                send_value(socket_client1, theta_0)
                # Send theta1 to second client
                send_value(socket_client2, theta_1)
                
                # Receive result from client 1
                result_client1 = socket_client1.recv(block=True)
                # Receive result from client 2
                result_client2 = socket_client2.recv(block=True)
                # put results into list
                qubit_results = [int(result_client1), int(result_client2)]
                
                # append to results the parity
                all_results[i][j] = check_parity(qubit_results)
                
                # calculate loss
                print(f"Calculated label: {all_results[i][j]}; Real label: {y[j]}")
                
                #TODO: Optimizer
                server.flush()
        print(all_results)
    # return dict of values
    return {
        "results": all_results.tolist(),
        "theta0": theta_0,
        "theta1": theta_1
    }
        
if __name__ == "__main__":
    main()
        



