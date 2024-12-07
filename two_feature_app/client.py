import math
import numpy as np
import utils.constants as constants
import os
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.build_types import HardwareConfig, GenericHardwareConfig
from utils.helper_functions import phase_gate, generate_chunks_with_max_size, split_array_by_nth_occurrences
from utils.socket_communication import receive_with_header, send_as_str, send_with_header, reset_socket
from utils.qubit_communication import remote_cnot_control, remote_cnot_target
from utils.logger import logger
from utils.feature_maps import ry_feature_map
from utils.config_parser import ConfigParser


class Client:
    def __init__(self, name: str, other_client_name: str, socket_id_with_server: int, socket_id_with_other_client: int, ctrl_qubit: bool):
        self.name = name
        self.socket_id_with_server = socket_id_with_server
        self.socket_id_with_other_client = socket_id_with_other_client
        self.socket_client = Socket(name, other_client_name, socket_id=socket_id_with_other_client)
        self.socket_server = Socket(self.name, "server", socket_id=self.socket_id_with_server)
        self.epr_socket_other_client = EPRSocket(remote_app_name=other_client_name, epr_socket_id=constants.EPR_C1_C2_C1, remote_epr_socket_id=constants.EPR_C1_C2_C2)
        self.ctrl_qubit = ctrl_qubit
        self.test_features = None
        self.params = None
        c = ConfigParser()
        self.max_qubits = c.max_qubits_per_qpu
        self.layers_with_rcnot = c.layers_with_rcnot


    def start_client(self):
        self.receive_starting_values_from_server()
        # receive instructions from server
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
        # receive nshots and qdepth
        self.params = receive_with_header(self.socket_server, constants.PARAMS)
        logger.info(f"{self.name} received params dict: {self.params}")
        
        # receive features
        self.features = receive_with_header(self.socket_server, constants.OWN_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received own features")
        
        # receive own test features
        self.test_features = receive_with_header(self.socket_server, constants.TEST_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received test features")
    
    
    def run_iteration(self, test=False):
        if test:
            features_array = self.test_features
        else:
            features_array = self.features
        
        # receive weights from server
        thetas = receive_with_header(self.socket_server, constants.THETAS, expected_dtype=np.ndarray)
        hw_config = GenericHardwareConfig(self.max_qubits)
        netqasm_connection = NetQASMConnection(
                app_name=self.name,
                epr_sockets=[self.epr_socket_other_client],
                max_qubits=self.max_qubits,
                hardware_config = hw_config
        )
        
        results = []
        with netqasm_connection:
            for i, features in enumerate(features_array):
                logger.debug(f"{self.name} Running feature number {i} with value(s) {features}")
                single_result = self.run_circuit_locally(features, thetas, netqasm_connection)
                results.append(single_result)
        results = np.array(results)
        send_with_header(self.socket_server, results, constants.RESULTS)

    
    def run_circuit_locally(self, features: np.array, weights: np.array, netqasm_connection: NetQASMConnection):
        '''
        Runs the PQC on this client. 
        This includes a feature map and an entanglement layer, that entangles with the other client as well.
        '''
        n_shots = self.params["n_shots"]
        q_depth = self.params["q_depth"]
        n_qubits = self.params["qubits_per_client"]
        
        # reshape weights
        weights_reshaped = np.split(weights, n_qubits)

        results_arr = np.empty((n_qubits,n_shots), dtype=np.int8)

        for i in range(n_shots):
            logger.debug(f"{self.name} is executing shot number {i+1} of {n_shots} shots")
            
            qubits = [Qubit(netqasm_connection) for _ in range(n_qubits)]
            
            # apply feature map
            for q, qbit in enumerate(qubits):
                ry_feature_map(qbit, features[q])
            
            # if we run out of EPR pairs, we generate new ones
            n_required_eprs = len(self.layers_with_rcnot) # number of required epr pairs is the amount of layers with a remote CNOT
            max_eprs = self.max_qubits - n_qubits # maximum value of EPR pairs that can be generated at once (due to hardware limitations)
            
            eprs = self.create_or_recv_epr_pairs(min(n_required_eprs, max_eprs), netqasm_connection) # generate first set of EPR pairs
            depth_epr_map = [1 if d in self.layers_with_rcnot else 0 for d in range(q_depth)] # map of which layers have an EPR pair
            
            for j, bit_val in enumerate(depth_epr_map):
                logger.debug(f"{self.name} entered layer: {j}")
                
                # rotate qubits with thetas
                for q, qbit in enumerate(qubits):
                    qbit.rot_Y(angle=weights_reshaped[q][j])

                    # apply local CNOT if it's not the "last" qubit of the node
                    if q < len(qubits) - 1:
                        qbit.cnot(qubits[q + 1])
                        
                    # if the RCNOT Map indicates a RCNOT, execute it (only for the "last qubit")
                    if q == len(qubits) - 1 and bit_val:
                        if not eprs: # if no EPRs are left, generate new ones
                            eprs = self.create_or_recv_epr_pairs(min(n_required_eprs, max_eprs), netqasm_connection)
                        epr = eprs.pop() # pop EPR pair
                        n_required_eprs -= 1 # reduce amount of required EPR pairs
                        if self.ctrl_qubit:
                            remote_cnot_control(self.socket_client, qbit, epr)
                        else:
                            remote_cnot_target(self.socket_client, qbit, epr)
          
            qubit_results = []
            # apply one more rotation in the end and measure
            for q, qbit in enumerate(qubits):
                qbit.rot_Y(angle=weights_reshaped[q][-1])
                qubit_results.append(qbit.measure())    
            
            netqasm_connection.flush()
            
            for q, result in enumerate(qubit_results):
                results_arr[q][i] = result.value
        return results_arr
    
            
    
    def create_or_recv_epr_pairs(self, n_pairs: int, netqasm_conn: NetQASMConnection):
        if self.ctrl_qubit:
            # create epr pairs
            assert self.socket_client.recv(block=True) == "ACK"
            netqasm_conn.flush()
            eprs = self.epr_socket_other_client.create_keep(number=n_pairs)
            logger.debug(f"{self.name} generated {n_pairs} epr pairs")
            
        else:
            # receive epr pairs
            self.socket_client.send("ACK")
            netqasm_conn.flush()
            eprs = self.epr_socket_other_client.recv_keep(number=n_pairs)
            logger.debug(f"{self.name} received {n_pairs} epr qubits")
        return eprs
    