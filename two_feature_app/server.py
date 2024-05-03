from typing import Literal, Union
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from utils.helper_functions import calculate_parity_from_shots, check_parity, prepare_dataset_iris, prepare_dataset_moons
from utils.socket_communication import send_with_header, receive_with_header
from scipy.optimize import minimize
import numpy as np
import math
import utils.constants as constants
from utils.logger import logger
from utils.plotting import plot_accs_and_losses

class QMLServer:
    def __init__(self, num_iter, initial_thetas, batch_size, learning_rate, random_seed, q_depth, n_shots, n_samples, test_size, dataset_function) -> None:
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.parameter_shift_delta = 0.001
        self.q_depth = q_depth
        self.n_qubits = 2
        self.n_shots = n_shots
        self.thetas = self.initialize_thetas(initial_thetas)
        
        # setup classical socket connections
        self.socket_client1 = Socket("server", "client1", socket_id=constants.SOCKET_SERVER_C1)
        self.socket_client2 = Socket("server", "client2", socket_id=constants.SOCKET_SERVER_C2)
        
        # setup EPR connections
        self.epr_socket_client1 = EPRSocket(remote_app_name="client1", epr_socket_id=constants.EPR_SERVER_C1_SERVER, remote_epr_socket_id=constants.EPR_SERVER_C1_C1)
        self.epr_socket_client2 = EPRSocket(remote_app_name="client2", epr_socket_id=constants.EPR_SERVER_C2_SERVER, remote_epr_socket_id=constants.EPR_SERVER_C2_C2)
        
        X, y = self.prepare_dataset(dataset_function, n_samples)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=test_size, random_state=random_seed, stratify=y)
        
        self.params = {
            'q_depth': q_depth,
            'n_shots': n_shots
        }
        self.server = NetQASMConnection(
            app_name="server",
            epr_sockets=[self.epr_socket_client1, self.epr_socket_client2],
        )
        self.iter_losses = []
        self.iter_accs = []
        
        
    def initialize_thetas(self, initial_thetas: list[Union[int, float]]) -> np.ndarray:
        # if no initial values initialize randomly
        if initial_thetas == None:
            initial_thetas = np.random.rand((self.q_depth + 1) * self.n_qubits)
        else:
            # convert to numpy float values
            initial_thetas = np.array(initial_thetas, dtype=float)
            assert len(initial_thetas) == (self.q_depth + 1) * self.n_qubits, "Not enough initial thetas provided"
        return initial_thetas
       
    
    def run_gradient_free(self, file_name: str) -> dict:
        iteration = 0
        # function to optimize
        # runs all data through our small network and computes the loss
        # returns the loss as the opitmization goal
        def method_to_optimize(params, ys):
            nonlocal iteration
            logger.info(f"Entering iteration {iteration}")
            iter_results = self.run_iteration(params)
            loss = self.calculate_loss(ys, iter_results)
            self.iter_losses.append(loss)
            acc = accuracy_score(ys, iter_results)
            self.iter_accs.append(acc)
            logger.info(f"Values in iteration {iteration}: Loss {loss}, Accuracy: {acc}")
            iteration += 1
            # prediction as iter results
            return loss
                
        # callback function executed after every iteration of the minimize function        
        def iteration_callback(intermediate_params):
            logger.info(f"Intermediate thetas: {intermediate_params}")
            
        with self.server:
            # send params and features to clients
            self.send_params_and_features()

            # minimize gradient free
            res = minimize(method_to_optimize, self.thetas, args=(self.y_train), options={'disp': True, 'maxiter': self.num_iter}, method="COBYLA", callback=iteration_callback)
            
            # test run
            dict_test_report = self.test_gradient_free()
            
            # exit clients
            self.send_exit_instructions()
            
            plot_accs_and_losses(file_name, self.iter_accs, self.iter_losses)
            
            return dict_test_report
    
    
    def test_gradient_free(self):
        test_results = self.run_iteration(self.thetas, test=True)
        # generate classification report
        dict_report = classification_report(y_true=self.y_test, y_pred=test_results, output_dict=True)
        print(dict_report)
        return dict_report
    
  
    def send_params_and_features(self):
        # send parameters to the clients
        send_with_header(self.socket_client1, self.params, constants.PARAMS)
        send_with_header(self.socket_client2, self.params, constants.PARAMS)
        
        features_client_1 = self.X_train[:, 0]
        features_client_2 = self.X_train[:, 1]
        
        # send their own features to the clients
        send_with_header(self.socket_client1, features_client_1, constants.OWN_FEATURES)
        send_with_header(self.socket_client2, features_client_2, constants.OWN_FEATURES)
        
        # send their counterparts features (because of ZZ Feature Map)
        send_with_header(self.socket_client1, features_client_2, constants.OTHER_FEATURES)
        send_with_header(self.socket_client2, features_client_1, constants.OTHER_FEATURES)
        
        
        # send the test features to the clients
        test_features_client_1 = self.X_test[:,0]
        test_features_client_2 = self.X_test[:,1]
        
        # send test features to the clients
        send_with_header(self.socket_client1, test_features_client_1, constants.TEST_FEATURES)
        send_with_header(self.socket_client2, test_features_client_2, constants.TEST_FEATURES)
        
        
  
    
    def run_iteration(self, params, test=False):
        if test:
            self.send_test_instructions()
        else:
            self.send_run_instructions()
        
        # split params array in half
        params_client_1 = params[:len(params)//2]
        params_client_2 = params[len(params)//2:]
        # Send thetas to first client
        send_with_header(self.socket_client1, params_client_1, constants.THETAS)
        # Send thetas to second client
        send_with_header(self.socket_client2, params_client_2, constants.THETAS)
        
        iter_results_client_1 = receive_with_header(self.socket_client1, constants.RESULTS)
        iter_results_client_2 = receive_with_header(self.socket_client2, constants.RESULTS)
        
        return self.calculate_iter_results(iter_results_client_1, iter_results_client_2)
        
    
    def calculate_iter_results(self, results_client_1: list[list[Literal[0,1]]], results_client_2: list[list[Literal[0,1]]]) -> list[int]:
        predicted_labels = []
        for i in range(len(results_client_1)):
            predicted_label = calculate_parity_from_shots(results_client_1[i], results_client_2[i])
            predicted_labels.append(predicted_label)
        return predicted_labels
    
    
    def calculate_loss(self, y_true, y_pred):
        loss = log_loss(y_true, y_pred, labels=[0,1])
        return loss
    
    
    def send_run_instructions(self):
        self.socket_client1.send(constants.RUN_INSTRUCTION)
        self.socket_client2.send(constants.RUN_INSTRUCTION)
        
    
    def send_test_instructions(self):
        self.socket_client1.send(constants.TEST_INSTRUCTION)
        self.socket_client2.send(constants.TEST_INSTRUCTION)
      
    
    def send_exit_instructions(self):
        self.socket_client1.send(constants.EXIT_INSTRUCTION)
        self.socket_client2.send(constants.EXIT_INSTRUCTION)
    

    def prepare_dataset(self, dataset: str, n_samples: int = 100):
        """
        Loads a dataset and returns it
        
        :param function: The function used to generate the dataset
        """
        if dataset.casefold() == "iris":
            return prepare_dataset_iris()
        elif dataset.casefold() == "moons":
            return prepare_dataset_moons(n_samples)
        else:
            raise ValueError("Inappropriate dataset provided: ", dataset)    
