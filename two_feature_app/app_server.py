from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, mean_squared_error
from helper_functions import send_value, check_parity, split_data_into_batches, send_dict, send_tensor
from scipy.optimize import minimize, OptimizeResult
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os

logger = get_netqasm_logger()


def main(app_config=None, num_iter=1, theta_initial_0=0, theta_initial_1=0, batch_size=1, learning_rate=0.01, random_seed=42):
    server = QMLServer(num_iter, [theta_initial_0, theta_initial_1], batch_size, learning_rate, random_seed)
    #server.run(server.process_batch_gradient_free, "batch_loss_gradient_free.png")
    server.run_gradient_free("batch_loss_gradient_free.png")
    return {
        "results": server.all_results.tolist(),
        "thetas": server.thetas
    }



class QMLServer:
    def __init__(self, num_iter, initial_thetas, batch_size, learning_rate, random_seed) -> None:
        self.num_iter = num_iter
        self.thetas = list(map(float, initial_thetas))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.parameter_shift_delta = 0.001
        
        # setup classical socket connections
        self.socket_client1 = Socket("server", "client1", socket_id=0)
        self.socket_client2 = Socket("server", "client2", socket_id=1)
        # setup EPR connections
        self.epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=0, remote_epr_socket_id=0)
        self.epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=1, remote_epr_socket_id=0)
        
        self.X, self.y = self.prepare_dataset()
        self.params = {
            'n_iters': self.num_iter,
            'n_samples': len(self.X),
            'batch_size': self.batch_size,
            'n_batches': math.ceil(len(self.X)/self.batch_size),
            'n_thetas': len(self.thetas)
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
                    for k in range(len(self.thetas)):
                        self.thetas[k] -= self.learning_rate * gradients[k]
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
                for k in range(len(self.thetas)):
                    self.thetas[k] -= self.learning_rate * gradients[k]
                # append to results
                iter_results = np.append(iter_results, batch_results)
                
                # add to all results
                self.all_results[i] = iter_results
        # send exit instruction to clients
        self.send_exit_instructions()
        self.plot_losses(file_name)
    
    
    def run_gradient_free(self, file_name):
        iteration = 0
        # function to optimize
        # runs all data through our small network and computes the loss
        # returns the loss as the opitmization goal
        def method_to_optimize(params, samples, ys):
            nonlocal iteration
            print(f"Entering iteration {iteration}")
            iter_results = np.empty(len(self.X))
            for i in range(len(samples)):
                iter_results[i] = self.run_circuits(samples[i], params)
            loss = self.calculate_loss(ys, iter_results)
            self.batch_losses.append(loss)
            #self.all_results[iteration] = iter_results
            print(f"Loss in iteration {iteration}: {loss}")
            iteration += 1
            # prediction as iter results
            return loss
                
        # callback function executed after every iteration of the minimize function        
        def iteration_callback(intermediate_params):
            print("Intermediate thetas: ", intermediate_params)
        
        with self.server:
            # send parameters to the clients
            send_dict(self.socket_client1, self.params)
            send_dict(self.socket_client2, self.params)
            
            # minimize gradient free
            res = minimize(method_to_optimize, self.thetas, args=(self.X, self.y), options={'disp': True, 'maxiter': self.num_iter}, method="nelder-mead", callback=iteration_callback)
            
            self.send_exit_instructions()
            self.plot_losses(file_name)

      
    def process_batch_gradient_free(self, batch, labels):
        batch_results = np.empty(len(batch))
        def method_to_optimize(params, batch, ys): 
            for i, sample in enumerate(batch):
                batch_results[i] = self.run_circuits(sample, params)
            loss = self.calculate_loss(ys, batch_results)
            return loss        
        res = minimize(method_to_optimize, self.thetas, args=(batch, labels), method="nelder-mead", options={'disp': True, 'maxiter': 1})
        self.thetas = res.x
        loss = res.fun
        gradients = np.zeros(len(self.thetas))
        return loss, gradients, batch_results

    
    
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
        
        # Send theta0 to first client
        send_value(self.socket_client1, params[0])
        # Send theta1 to second client
        send_value(self.socket_client2, params[1])
        
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
    
    
    def prepare_dataset(self):
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

    
if __name__ == "__main__":
    main()
        



