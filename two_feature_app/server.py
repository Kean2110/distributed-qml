import os
import time
from typing import Literal, Union
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from utils.config_parser import ConfigParser
from utils.helper_functions import calculate_parity_from_shots, check_parity, prepare_dataset_iris, prepare_dataset_moons, load_latest_checkpoint
from utils.model_saver import ModelSaver
from utils.socket_communication import send_with_header, receive_with_header
from scipy.optimize import minimize, Bounds
from utils.timer import global_timer
import numpy as np
import math
import utils.constants as constants
from utils.logger import logger
from utils.plotting import plot_accs_and_losses
from netqasm.sdk.shared_memory import SharedMemoryManager

class QMLServer:
    def __init__(self, n_qubits, max_iter, initial_thetas, random_seed, q_depth, n_shots, n_samples, test_size, dataset_function, start_from_checkpoint, output_path, test_data=None) -> None:
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.parameter_shift_delta = 0.001
        self.q_depth = q_depth
        self.n_qubits = 2
        self.n_shots = n_shots
        self.output_path = output_path
        # load the params from the checkpoint
        self.load_params_from_checkpoint(start_from_checkpoint)
        self.thetas = initial_thetas if self.thetas is None else self.thetas # covers the case that we are first iteration in our checkpoint iter, so we return None as thetas and want to still have our initial params that we defined in the config
        self.thetas = self.initialize_thetas(self.thetas)
        
        # setup classical socket connections
        self.socket_client1 = Socket("server", "client1", socket_id=constants.SOCKET_SERVER_C1)
        self.socket_client2 = Socket("server", "client2", socket_id=constants.SOCKET_SERVER_C2)
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_dataset(dataset_function, n_samples, test_size, random_seed, test_data)
        
        # initialize the model saver
        # if we didnt load losses from the checkpoint we dont have a best loss yet
        # else it's the last loss of our losses
        best_loss = None if not self.iter_losses else self.iter_losses[-1]
        self.ms = ModelSaver(self.output_path, best_loss)
        
    
    def load_params_from_checkpoint(self, start_from_checkpoint):
        checkpoint_path = os.path.join(self.output_path, "checkpoints")
        if start_from_checkpoint:
            self.thetas, self.start_iteration, self.iter_losses, self.iter_accs = load_latest_checkpoint(checkpoint_path)
            logger.info(f"Loaded params {self.thetas}, iteration no {self.start_iteration}, losses {self.iter_losses} and accs {self.iter_accs} from checkpoint")
            self.max_iter -= self.start_iteration
        else:
            self.start_iteration = 0
            self.iter_losses = []
            self.iter_accs = []
            self.thetas = None
    
        
    def initialize_thetas(self, initial_thetas: list[Union[int, float]]) -> np.ndarray:
        # if no initial values initialize randomly
        if initial_thetas is None:
            initial_thetas = np.random.rand((self.q_depth + 1) * self.n_qubits)
        else:
            # convert to numpy float values
            initial_thetas = np.array(initial_thetas, dtype=float)
        assert len(initial_thetas) == (self.q_depth + 1) * self.n_qubits, "Not enough initial thetas provided"
        return initial_thetas
       
    @global_timer.timer
    def run_gradient_free(self, file_name: str) -> dict:
        iteration = self.start_iteration
        
        # function to optimize
        # runs all data through our small network and computes the loss
        # returns the loss as the opitmization goal
        def method_to_optimize(params, ys):
            nonlocal iteration
            logger.info(f"Entering iteration {iteration + 1} of {self.max_iter + self.start_iteration}")
            # run the model 
            start_time = time.time()
            iter_results = self.run_iteration(params)
            SharedMemoryManager.reset_memories() # reset memories between clients and the QuantumNodes in order to reduce memory consumption after each iteration
            end_time = time.time()
            diff_time_mins = (end_time - start_time)/60.0
            # calculate the loss
            loss = self.calculate_loss(ys, iter_results)
            # save results
            self.iter_losses.append(loss)
            # calculate accuracy
            acc = accuracy_score(ys, iter_results)
            self.iter_accs.append(acc)
            logger.info(f"Values in iteration {iteration + 1}: Loss {loss}, Accuracy: {acc}, Elpased Minutes: {diff_time_mins}")
            # count up iteration
            iteration += 1
            # save params with modelsaver
            self.ms.save_intermediate_results(params, iteration, self.iter_losses, self.iter_accs)
            return loss
                
        # callback function executed after every iteration of the minimize function        
        def iteration_callback(intermediate_params):
            logger.debug(f"Intermediate thetas: {intermediate_params}")
            
        c = ConfigParser()
        logger.info(c.get_config())
        # send params and features to clients
        self.send_params_and_features()

        # minimize gradient free
        theta_bounds = Bounds(0, 2*math.pi)
        res = minimize(method_to_optimize, self.thetas, args=(self.y_train), options={'disp': True, 'maxiter': self.max_iter}, method="COBYLA", bounds=theta_bounds, callback=iteration_callback)
        # test run
        dict_test_report = self.test_gradient_free()
        
        # exit clients
        self.send_exit_instructions()
            
        plot_accs_and_losses(file_name, self.output_path, self.iter_accs, self.iter_losses)
        
        return dict_test_report
    
    
    def test_gradient_free(self):
        test_results = self.run_iteration(self.thetas, test=True)
        # generate classification report
        dict_report = classification_report(y_true=self.y_test, y_pred=test_results, output_dict=True)
        print(dict_report)
        return dict_report
    
  
    def send_params_and_features(self):
        # create params dict
        params_dict = {"n_shots": self.n_shots, "q_depth": self.q_depth}
        # send params
        send_with_header(self.socket_client1, params_dict, constants.PARAMS)
        send_with_header(self.socket_client2, params_dict, constants.PARAMS)
        
        # split up features for clients 1 and 2
        # if we have only test features, this is set to None
        features_client_1 = self.X_train[:, 0] if self.X_train is not None else None
        features_client_2 = self.X_train[:, 1] if self.X_train is not None else None
            
        # send their own features to the clients
        send_with_header(self.socket_client1, features_client_1, constants.OWN_FEATURES)
        send_with_header(self.socket_client2, features_client_2, constants.OWN_FEATURES)
        
        # send their counterparts features (because of ZZ Feature Map)
        send_with_header(self.socket_client1, features_client_2, constants.OTHER_FEATURES)
        send_with_header(self.socket_client2, features_client_1, constants.OTHER_FEATURES)
        
        # split up test features
        test_features_client_1 = self.X_test[:,0]
        test_features_client_2 = self.X_test[:,1]
        
        # send test features to the clients
        send_with_header(self.socket_client1, test_features_client_1, constants.TEST_FEATURES)
        send_with_header(self.socket_client2, test_features_client_2, constants.TEST_FEATURES)

    
    @global_timer.timer
    def run_iteration(self, params, test=False):
        if test:
            self.send_test_instructions()
        else:
            self.send_run_instructions()
        
        # split params array in half
        # params look like [client1, client2, client1, client2, ....]
        params_client_1 = params[::2]
        params_client_2 = params[1::2]
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
    

    def prepare_dataset(self, dataset_name: str, n_samples: int = 100, test_size: float = 0.2, random_seed : int = 42, test_dataset = None):
        """
        Loads a dataset and returns the split into test and train data.
        In case a test dataset is provided, only split up into data and labels
        
        :param function: The function used to generate the dataset
        :returns X_train, X_test, y_train, y_test
        :raises ValueError if the dataset string is inappropriate.
        """
        # if we want only test data, simply split up the test dataset
        if test_dataset:
            return None, test_dataset["data"], None, test_dataset["labels"]
        
        if dataset_name.casefold() == "iris":
            X, y = prepare_dataset_iris()
            return train_test_split(X,y, test_size=test_size, random_state=random_seed, stratify=y)
        elif dataset_name.casefold() == "moons":
            X, y = prepare_dataset_moons(n_samples)
            return train_test_split(X,y, test_size=test_size, random_state=random_seed, stratify=y)
        else:
            raise ValueError("Inappropriate dataset provided: ", dataset_name)    
