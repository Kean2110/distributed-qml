from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
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


def calculate_gradient(loss):
    return loss

def process_batch(batch, server, sockets, thetas, labels):
    # In this way we can reuse the code for the first batches and the last smaller one
    batch_results = np.empty(len(batch))
    for j,sample in enumerate(batch):
        # Send first feature to client 1
        send_value(sockets[0], sample[0])
        # Send second feature to client 2
        send_value(sockets[1], sample[1])
        
        # Send theta0 to first client
        send_value(sockets[0], thetas[0])
        # Send theta1 to second client
        send_value(sockets[1], thetas[1])
        
        # Receive result from client 1
        result_client1 = sockets[0].recv(block=True)
        # Receive result from client 2
        result_client2 = sockets[1].recv(block=True)
        # put results into list
        qubit_results = [int(result_client1), int(result_client2)]
        
        # append to results the parity
        batch_results[j] = check_parity(qubit_results)
        
        # calculate loss
        print(f"Calculated label: {batch_results[j]}; Real label: {labels[j]}")
        
    server.flush()
    loss = log_loss(y_true=labels, y_pred=batch_results, labels=[0,1])
        
    # TODO: gradient calculation
    gradient = calculate_gradient(loss)
    
    return loss, gradient, batch_results
    
    
def main(app_config=None, num_iter=1, theta_initial_0=0, theta_initial_1=0, batch_size=1, learning_rate=0.01):
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
    
    # initialize weights
    theta_0 = theta_initial_0
    theta_1 = theta_initial_1

    batches = split_data_into_batches(X, batch_size)
    
    with server:
        
        # send parameters to the clients
        params = {'n_iters': num_iter, 'n_samples': len(X), 'batch_size': batch_size, 'n_batches': len(batches)}
        send_dict(socket_client1, params)
        send_dict(socket_client2, params)
        
        all_results = np.empty((num_iter, len(X)))
        for i in range(num_iter):
            print(f"ENTERED ITERATION {i+1}")
            iter_results = np.array([])
            for b in range(len(batches) -1):
                labels = y[b*batch_size:b*batch_size+batch_size]
                loss, gradient, batch_results = process_batch(batches[b], server, [socket_client1, socket_client2], [theta_0, theta_1], labels)
                print(f"Calculated loss for iteration {i+1}, batch {b+1}: {loss}")
                theta_0 -= learning_rate * gradient
                theta_1 -= learning_rate * gradient
                iter_results = np.append(iter_results, batch_results)
            # process last_batch seperately, because it might have a different length
            last_batch = batches[-1]
            # get last few elements as labels
            labels = y[-len(last_batch):]
            loss, gradient, batch_results = process_batch(last_batch, server, [socket_client1, socket_client2], [theta_0, theta_1], labels)
            print(f"Calculated loss for iteration {i+1}, batch {len(batches) + 1}: {loss}")
            theta_0 -= learning_rate * gradient
            theta_1 -= learning_rate * gradient
            iter_results = np.append(iter_results, batch_results)
            
            # append to all results
            all_results[i] = iter_results
        print(all_results)
    # return dict of values
    return {
        "results": all_results.tolist(),
        "theta0": theta_0,
        "theta1": theta_1
    }
        
if __name__ == "__main__":
    main()
        



