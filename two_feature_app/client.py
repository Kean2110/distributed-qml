import math

import numpy as np
import utils.constants as constants
import logging
from logging import Logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from utils.helper_functions import phase_gate
from utils.socket_communication import receive_with_header, send_as_str, send_with_header
from utils.qubit_communication import remote_cnot_control, remote_cnot_target
from utils.logger import logger
from utils.config_parser import ConfigParser


class Client:
    def __init__(self, name: str, other_client_name: str, socket_id_with_server: int, socket_id_with_other_client: int, epr_socket_id_server: int, ctrl_qubit: bool, n_shots: int, q_depth: int):
        self.name = name
        self.socket_server = Socket(name, "server", socket_id=socket_id_with_server)
        self.socket_client = Socket(name, other_client_name, socket_id=socket_id_with_other_client)
        self.epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=constants.EPR_SERVER_C1_C1, remote_epr_socket_id=epr_socket_id_server)
        self.epr_socket_other_client = EPRSocket(remote_app_name=other_client_name, epr_socket_id=constants.EPR_C1_C2_C1, remote_epr_socket_id=constants.EPR_C1_C2_C2)
        self.ctrl_qubit = ctrl_qubit
        self.eprs_needed_for_feature_map = 0
        self.features = None
        self.features_other_node = None
        self.test_features = None
        self.n_shots = n_shots
        self.q_depth = q_depth
        
        self.netqasm_connection = NetQASMConnection(
            app_name=name,
            epr_sockets=[self.epr_socket_server, self.epr_socket_other_client],
        )



    def start_client(self):
        self.receive_starting_values_from_server()
        # receive instructions from server
        with self.netqasm_connection:
            while True:
                instruction = self.socket_server.recv(block=True)
                if instruction == constants.EXIT_INSTRUCTION:
                    break
                elif instruction == constants.RUN_INSTRUCTION:
                    self.run_iteration()
                elif instruction == constants.TEST_INSTRUCTION:
                    self.run_iteration(test=True)
                else:
                    raise ValueError("Unregistered instruction received")
    

    def receive_starting_values_from_server(self):
        self.features = receive_with_header(self.socket_server, constants.OWN_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received own features")
        
        self.features_other_node = receive_with_header(self.socket_server, constants.OTHER_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received other nodes features")
        
        self.test_features = receive_with_header(self.socket_server, constants.TEST_FEATURES)
        logger.info(f"{self.name} received test features")
    
    
    def run_iteration(self, test=False):
        if test:
            features = self.test_features
        else:
            features = self.features
        # receive thetas from server
        thetas = receive_with_header(self.socket_server, constants.THETAS, expected_dtype=np.ndarray)
        results = []
        for i, feature in enumerate(features):
            logger.debug(f"{self.name} Running feature number {i}")
            single_result = self.run_circuit_locally(feature, thetas)
            results.append(single_result)
        
        send_with_header(self.socket_server, results, constants.RESULTS)

    
    #@timer
    def run_circuit_locally(self, feature: float, weights: list[float]):
        
        results_arr = []
        
        for i in range(self.n_shots):
            logger.debug(f"{self.name} is executing shot number {i+1} of {self.n_shots} shots")
            
            eprs = self.create_or_recv_epr_pairs(self.q_depth + self.eprs_needed_for_feature_map)
            
            q = Qubit(self.netqasm_connection)
            
            self.ry_feature_map(q, feature)
            
            eprs = eprs[self.eprs_needed_for_feature_map:]
            
            # execute hidden layer
            for j in range(self.q_depth):
                logger.debug(f"{self.name} entered layer: {j+1}")
                
                # apply theta rotation
                q.rot_Y(angle=weights[j])
                if self.ctrl_qubit:
                    remote_cnot_control(self.socket_client, q, eprs[j])
                else:
                    remote_cnot_target(self.socket_client, q, eprs[j])
            q.rot_Y(angle=weights[-1])        
            
            # measure
            result = q.measure()
            
            self.netqasm_connection.flush()
            results_arr.append(result.value)
            
            
        return results_arr
            
    
    def create_or_recv_epr_pairs(self, n_pairs: int):
        if self.ctrl_qubit:
            # create epr pairs
            eprs = self.epr_socket_other_client.create_keep(number=n_pairs)
            logger.debug(f"{self.name} generated {n_pairs} epr pairs")
        else:
            # receive epr pairs
            eprs = self.epr_socket_other_client.recv_keep(number=n_pairs)
            logger.debug(f"{self.name} received {n_pairs} epr qubits")
        return eprs

    
    def zz_feature_map_ctrl(self, qubit: Qubit, eprs: list[Qubit], feature: float):
        qubit.H()
        phase_gate(2 * feature, qubit)
        remote_cnot_control(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        remote_cnot_control(self.socket_client, self.netqasm_connection, qubit, eprs[1])
        logger.debug(f"{self.name} executed zz feature map control")
        
    
    def zz_feature_map_target(self, qubit: Qubit, eprs: list[Qubit], feature: float, feature_other_node: list[float]):
        qubit.H()
        phase_gate(angle = 2 * feature, qubit = qubit)
        remote_cnot_target(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        phase_gate(angle = 2 * (math.pi - feature_other_node) * (math.pi - feature), qubit = qubit)
        remote_cnot_target(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        logger.debug(f"{self.name} executed zz feature map target")
        
        
    def ry_feature_map(self, qubit: Qubit, feature: float):
        qubit.rot_Y(angle=feature)