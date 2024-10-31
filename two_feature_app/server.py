import os
import time
from typing import Literal, Union
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from utils.config_parser import ConfigParser
from utils.helper_functions import calculate_parity_from_shots, check_parity, prepare_dataset_iris, prepare_dataset_moons, load_latest_checkpoint, lower_bound_constraint, upper_bound_constraint, split_data_into_batches
from utils.model_saver import ModelSaver
from utils.socket_communication import send_with_header, receive_with_header
from scipy.optimize import minimize
from utils.timer import global_timer
import numpy as np
import math
import utils.constants as constants
from utils.logger import logger
from utils.plotting import plot_accs_and_losses
from netqasm.sdk.shared_memory import SharedMemoryManager

class QMLServer:
    def __init__(self, n_qubits, epochs, initial_thetas, random_seed, q_depth, n_shots, n_samples, test_size, dataset_function, start_from_checkpoint, output_path, batch_size, test_data=None) -> None:
        self.n_qubits = n_qubits
        self.random_seed = random_seed
        self.parameter_shift_delta = 0.001
        self.q_depth = q_depth
        self.n_shots = n_shots
        self.batch_size = batch_size
        self.output_path = output_path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_batched, self.y_train_batched = None, None
        self.socket_client1, self.socket_client2 = None, None
        self.n_batches, self.iterations = None, None
        self.start_iteration = 0
        self.iter_losses, self.iter_accs = [], []
        self.thetas = initial_thetas
        
        # setup the dataset
        self.setup_dataset(epochs, dataset_function, n_samples, test_size, random_seed, n_qubits, test_data)
        
        # load the params from the checkpoint
        if start_from_checkpoint:
            self.load_params_from_checkpoint()
        self.thetas = self.initialize_thetas(self.thetas)
        
        self.setup_sockets()
        self.setup_model_saver()
    
    
    def setup_dataset(self, epochs, dataset_function, n_samples, test_size, random_seed, n_features, test_data):
        # load data and initialize iterations accordingly
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_dataset(dataset_function, n_samples, n_features, test_size, random_seed, test_data)
        self.X_train_batched, self.y_train_batched = split_data_into_batches(self.X_train, self.y_train, self.batch_size)
        self.n_batches = len(self.y_train_batched)
        self.iterations = epochs * self.n_batches # iterations = epochs * number of batches
        
    
    def setup_sockets(self):
        # setup classical socket connections
        self.socket_client1 = Socket("server", "client1", socket_id=constants.SOCKET_SERVER_C1)
        self.socket_client2 = Socket("server", "client2", socket_id=constants.SOCKET_SERVER_C2)
        
    
    def setup_model_saver(self):
        # initialize the model saver
        # if we didnt load losses from the checkpoint we dont have a best loss yet
        # else it's the last loss of our losses
        best_loss = None if not self.iter_losses else self.iter_losses[-1]
        self.ms = ModelSaver(self.output_path, best_loss)
    
    
    def load_params_from_checkpoint(self):
        checkpoint_path = os.path.join(self.output_path, "checkpoints")
        self.thetas, self.start_iteration, self.iter_losses, self.iter_accs = load_latest_checkpoint(checkpoint_path)
        logger.info(f"Loaded params {self.thetas}, iteration no {self.start_iteration}, losses {self.iter_losses} and accs {self.iter_accs} from checkpoint")
        self.iterations -= self.start_iteration
    
        
    def initialize_thetas(self, initial_thetas: list[Union[int, float]]) -> np.ndarray:
        # if no initial values initialize randomly
        if initial_thetas is None:
            #initial_thetas = np.random.rand((self.q_depth + 1) * self.n_qubits)
            np.random.seed(self.random_seed)
            initial_thetas = np.random.uniform(constants.LOWER_BOUND_PARAMS, constants.UPPER_BOUND_PARAMS, (self.q_depth + 1) * self.n_qubits)
        else:
            # convert to numpy float values
            initial_thetas = np.array(initial_thetas, dtype=float)
        assert len(initial_thetas) == (self.q_depth + 1) * self.n_qubits, "Not enough initial thetas provided"
        return initial_thetas
       
    @global_timer.timer
    def train_gradient_free(self, file_name: str) -> dict:
        iteration = self.start_iteration
        
        # function to optimize
        # runs all data through our small network and computes the loss
        # returns the loss as the opitmization goal
        def method_to_optimize(params):
            nonlocal iteration
            logger.info(f"Entering iteration {iteration + 1} of {self.iterations + self.start_iteration}")
            # run the model 
            start_time = time.time()
            # get current batch
            features = self.X_train_batched[iteration % self.n_batches]
            labels = self.y_train_batched[iteration % self.n_batches]
            # run iteration on clients
            iter_results = self.run_iteration(params, features)
            SharedMemoryManager.reset_memories() # reset memories between clients and the QuantumNodes in order to reduce memory consumption after each iteration
            end_time = time.time()
            diff_time_mins = (end_time - start_time)/60.0
            # calculate the loss
            loss = self.calculate_loss(labels, iter_results)
            # save results
            self.iter_losses.append(loss)
            # calculate accuracy
            acc = accuracy_score(labels, iter_results)
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
        self.send_params_and_test_features()

        # minimize gradient free
        constraints = [
            {'type': 'ineq', 'fun': lower_bound_constraint},
            {'type': 'ineq', 'fun': upper_bound_constraint}
        ]
        
        res = minimize(method_to_optimize, self.thetas, options={'disp': True, 'maxiter': self.iterations}, method="COBYLA", constraints=constraints, callback=iteration_callback)
        # test run
        dict_test_report = self.test_gradient_free()
        
        # exit clients
        self.send_exit_instructions()
            
        plot_accs_and_losses(file_name, self.output_path, self.iter_accs, self.iter_losses)
        
        return dict_test_report
    
    
    def test_gradient_free(self):
        test_results = self.run_iteration(self.thetas, None, test=True)
        # generate classification report
        dict_report = classification_report(y_true=self.y_test, y_pred=test_results, output_dict=True)
        print(dict_report)
        return dict_report
    
    
    def send_params_and_test_features(self):
        # create params dict
        params_dict = {"n_shots": self.n_shots, "q_depth": self.q_depth, "qubits_per_client": self.n_qubits // 2}
        # send params
        send_with_header(self.socket_client1, params_dict, constants.PARAMS)
        send_with_header(self.socket_client2, params_dict, constants.PARAMS)
        
        # split up test features
        test_features_client_1, test_features_client_2 = np.hsplit(self.X_test, 2)
        
        # send test features to the clients
        send_with_header(self.socket_client1, test_features_client_1, constants.TEST_FEATURES)
        send_with_header(self.socket_client2, test_features_client_2, constants.TEST_FEATURES)

    
    def send_features(self, features):
        features_client_1, features_client_2 = np.hsplit(features, 2)
        
        # send their own features to the clients
        send_with_header(self.socket_client1, features_client_1, constants.OWN_FEATURES)
        send_with_header(self.socket_client2, features_client_2, constants.OWN_FEATURES)

    
    @global_timer.timer
    def run_iteration(self, params, features, test=False):
        if test:
            self.send_test_instructions()
        else:
            self.send_run_instructions()
            self.send_features(features)
        
        # split params array in half
        params_client_1, params_client_2 = np.hsplit(params, 2)
        # Send thetas to first client
        send_with_header(self.socket_client1, params_client_1, constants.THETAS)
        # Send thetas to second client
        send_with_header(self.socket_client2, params_client_2, constants.THETAS)
        
        iter_results_client_1 = receive_with_header(self.socket_client1, constants.RESULTS)
        iter_results_client_2 = receive_with_header(self.socket_client2, constants.RESULTS)
        
        return self.calculate_iter_results(iter_results_client_1, iter_results_client_2)
        
    
    def calculate_iter_results(self, results_client_1: list[list[list[Literal[0,1]]]], results_client_2: list[list[list[Literal[0,1]]]]) -> list[int]:
        '''
        Calculate the results per iteration of both clients.
        Results_client_X consists of a list that is n_features * n_qubits * n_shots and is a list of 0 and 1s, which are the measurement results of the qubit
        '''
        # results_client_X look like n_features * n_qubits * n_shots
        predicted_labels_per_feature = []
        # iterate over features
        for i in range(len(results_client_1)):
            # concatenate both lists to get all qubit results
            feature_results_split_into_qubits = results_client_1[i] + results_client_2[i]
            predicted_label = calculate_parity_from_shots(feature_results_split_into_qubits)
            predicted_labels_per_feature.append(predicted_label)
        return predicted_labels_per_feature
    
    
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
    

    def prepare_dataset(self, dataset_name: str, n_samples: int = 100, n_features: int = 2, test_size: float = 0.2, random_seed : int = 42, test_dataset = None):
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
            X, y = prepare_dataset_iris(n_features)
            return train_test_split(X,y, test_size=test_size, random_state=random_seed, stratify=y)
        elif dataset_name.casefold() == "moons":
            if n_features != 2:
                raise ValueError("The Moons dataset only supports two features")
            X, y = prepare_dataset_moons(n_samples)
            return train_test_split(X,y, test_size=test_size, random_state=random_seed, stratify=y)
        else:
            raise ValueError("Inappropriate dataset provided: ", dataset_name)    
