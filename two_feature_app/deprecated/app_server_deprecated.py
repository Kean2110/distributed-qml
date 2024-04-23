from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, mean_squared_error
from utils.helper_functions import send_value, check_parity, split_data_into_batches, send_dict, send_tensor, prepare_dataset_iris, prepare_dataset_moons, send_with_header
from scipy.optimize import minimize, OptimizeResult
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import utils.constants as constants

logger = get_netqasm_logger()


def main(app_config=None, num_iter=1, initial_thetas_client_1=None, initial_thetas_client_2=None, batch_size=1, learning_rate=0.01, random_seed=42, q_depth=1, dataset_function="iris"):
    server = QMLServer(num_iter, [initial_thetas_client_1, initial_thetas_client_2], batch_size, learning_rate, random_seed, q_depth, dataset_function)
    try: 
        #server.run(server.process_batch_gradient_free, "batch_loss_gradient_free.png")
        server.run_gradient_free("batch_loss_gradient_free.png")
    except Exception as e:
        print("An error occured in server: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
        
    return {
        "results": server.all_results.tolist(),
        "thetas": server.thetas
    }



class QMLServer:
    def __init__(self, num_iter, initial_thetas, batch_size, learning_rate, random_seed, q_depth, dataset_function) -> None:
        self.num_iter = num_iter
        # if no initial values initialize randomly
        for i, initial_theta_list in enumerate(initial_thetas):
            if initial_theta_list == None:
                initial_theta_list = [random.random() for _ in range(q_depth)]
            # assert that there are enough parameters
            assert len(initial_theta_list) == q_depth, "Not enough initial theta values provided"
            # convert to float values
            initial_theta_list = list(map(float, initial_theta_list))
            initial_thetas[i] = initial_theta_list
        # flatten list and convert to numpy arrays
        self.thetas = np.array(initial_thetas)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.parameter_shift_delta = 0.001
        self.q_depth = q_depth
        
        # setup classical socket connections
        self.socket_client1 = Socket("server", "client1", socket_id=0)
        self.socket_client2 = Socket("server", "client2", socket_id=1)
        # setup EPR connections
        self.epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=0, remote_epr_socket_id=0)
        self.epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=1, remote_epr_socket_id=0)
        
        self.X, self.y = self.prepare_dataset(dataset_function)
        self.params = {
            'n_iters': self.num_iter,
            'n_samples': len(self.X),
            'batch_size': self.batch_size,
            'n_batches': math.ceil(len(self.X)/self.batch_size),
            'n_thetas': len(self.thetas[0]),
            'q_depth': q_depth
        }
        self.all_results = np.empty((num_iter, len(self.X)))
        self.server = NetQASMConnection(
            app_name="server",
            epr_sockets=[self.epr_socket_client1, self.epr_socket_client2],
        )
        self.batch_losses = []
    
    
    def run(self, batch_function, file_name):
        with self.server:
            # send parameters to the clients
            
            send_dict(self.socket_client1, self.params)
            send_dict(self.socket_client2, self.params)

            for i in range(self.num_iter):
                print(f"ENTERED ITERATION {i+1}")
                # in the loop so the smaller batch always contains differnet elements
                batches, labels = split_data_into_batches(self.X, self.y, self.batch_size, random_seed=self.random_seed)
                iter_results = np.array([])
                # process all batches except the last one
                for j in range(len(batches) -1):
                    loss, gradients, batch_results = batch_function(batches[j], labels[j])
                    print(f"Calculated loss for iteration {i+1}, batch {j+1}: {loss}")
                    # recalculate weights
                    for k in range(len(self.thetas[0])):
                        for l, client_thetas in enumerate(self.thetas):
                            client_thetas[k] -= self.learning_rate * gradients[l][k]
                    iter_results = np.append(iter_results, batch_results)
                    self.batch_losses.append(loss)
                    
                # calculate last_batch seperately, because it might have a different length
                last_batch = batches[-1]
                last_labels = labels[-1]
                # process the last batch
                loss, gradients, batch_results = batch_function(last_batch, last_labels)
                print(f"Calculated loss for iteration {i+1}, batch {len(batches)}: {loss}")
                self.batch_losses.append(loss)
                # recalculate weights
                for k in range(len(self.thetas[0])):
                        for l, client_thetas in enumerate(self.thetas):
                            client_thetas[k] -= self.learning_rate * gradients[l][k]
                # append to results
                iter_results = np.append(iter_results, batch_results)
                
                # add to all results
                self.all_results[i] = iter_results
        # send exit instruction to clients
        self.send_exit_instructions()
        self.plot_losses(file_name)
    
    
    def process_batch_param_shift(self, batch, labels):
        # In this way we can reuse the code for the first batches and the last smaller one
        batch_results = np.empty(len(batch))
        
        # used for gradient calculation with parameter shift
        batch_results_plus = np.empty((len(batch), len(self.thetas)))
        batch_results_minus = np.empty((len(batch), len(self.thetas)))
        
        gradients = np.empty(len(self.thetas), float)
        for i,sample in enumerate(batch):
            # run the circuit
            batch_results[i] = self.run_circuits(sample, self.thetas)
            
            # calculate gradients through parameter shift
            for j in range(len(self.thetas)):
                # copy the params to tweak them
                thetas_plus = self.thetas.copy()
                thetas_minus = self.thetas.copy()
                
                thetas_plus[j] += self.parameter_shift_delta
                thetas_minus[j] -= self.parameter_shift_delta
                
                batch_results_plus[i][j] = self.run_circuits(sample, thetas_plus)
                batch_results_minus[i][j] = self.run_circuits(sample, thetas_minus)     
        
        # calculate losses
        for j in range(len(self.thetas)):    
            plus_loss = self.calculate_loss(labels, batch_results_plus[:,j])
            minus_loss = self.calculate_loss(labels, batch_results_minus[:,j])
            gradient = (plus_loss - minus_loss) / (2 * self.parameter_shift_delta)
            gradients[j] = gradient
        loss = self.calculate_loss(labels, batch_results)
        
        return loss, gradients, batch_results
                     
        
    def run_circuits(self, features, params):
        self.send_run_instructions()
        
        # Send first feature to client 1
        send_value(self.socket_client1, features[0])
        # Send second feature to client 2
        send_value(self.socket_client2, features[1])
        
        # split params array in half
        params_client_1 = params[:len(params)//2]
        params_client_2 = params[len(params)//2:]
        for i in range(self.q_depth):
            # Send theta0 to first client
            send_value(self.socket_client1, params_client_1[i])
            # Send theta1 to second client
            send_value(self.socket_client2, params_client_2[i])
        
        # Receive result from client 1
        result_client1 = self.socket_client1.recv(block=True)
        # Receive result from client 2
        result_client2 = self.socket_client2.recv(block=True)
        # put results into list
        qubit_results = [int(result_client1), int(result_client2)]
        
        # append to results the parity
        predicted_label = check_parity(qubit_results)
        
        return predicted_label
    
    
    def calculate_loss(self, y_true, y_pred):
        loss = log_loss(y_true, y_pred, labels=[0,1])
        return loss
    
    
    def send_run_instructions(self):
        self.socket_client1.send("RUN CIRCUIT")
        self.socket_client2.send("RUN CIRCUIT")
        
    
    def send_exit_instructions(self):
        self.socket_client1.send("EXIT")
        self.socket_client2.send("EXIT")
    
    
    def plot_losses(self, filename: str) -> None:
        plt.plot(self.batch_losses)
        plt.xlabel("batch number")
        plt.ylabel("log loss")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_directory = os.path.join(script_dir, "plots")
        plt.savefig(os.path.join(plot_directory, filename))
    
    
    def prepare_dataset(self, dataset: str):
        """
        Loads a dataset and returns it
        
        :param function: The function used to generate the dataset
        """
        if dataset.casefold() == "iris":
            return prepare_dataset_iris()
        elif dataset.casefold() == "moons":
            return prepare_dataset_moons()
        else:
            raise ValueError("Inappropriate dataset provided: ", dataset)    

    
if __name__ == "__main__":
    main()
        



