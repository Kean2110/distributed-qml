import math
import numpy as np
import utils.constants as constants
import os
#os.environ["NETQASM_SIMULATOR"] = "netsquid_single_thread"
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket, Qubit
from utils.helper_functions import phase_gate, generate_chunks_with_max_size
from utils.socket_communication import receive_with_header, send_as_str, send_with_header
from utils.qubit_communication import remote_cnot_control, remote_cnot_target
from utils.logger import logger
from utils.feature_maps import ry_feature_map


class Client:
    def __init__(self, name: str, other_client_name: str, socket_id_with_server: int, socket_id_with_other_client: int, epr_socket_id_server: int, ctrl_qubit: bool):
        self.name = name
        self.socket_server = Socket(name, "server", socket_id=socket_id_with_server)
        self.socket_client = Socket(name, other_client_name, socket_id=socket_id_with_other_client)
        self.epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=constants.EPR_SERVER_C1_C1, remote_epr_socket_id=epr_socket_id_server)
        self.epr_socket_other_client = EPRSocket(remote_app_name=other_client_name, epr_socket_id=constants.EPR_C1_C2_C1, remote_epr_socket_id=constants.EPR_C1_C2_C2)
        self.ctrl_qubit = ctrl_qubit
        self.features = None
        self.features_other_node = None
        self.test_features = None
        self.params = None
        max_qubits = constants.MAX_VALUES["eprs"] + 1
        self.netqasm_connection = NetQASMConnection(
            app_name=name,
            epr_sockets=[self.epr_socket_server, self.epr_socket_other_client],
            max_qubits=max_qubits
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
        # receive nshots and qdepth
        self.params = receive_with_header(self.socket_server, constants.PARAMS)
        logger.info(f"{self.name} received params dict: {self.params}")
        
        # receive features
        self.features = receive_with_header(self.socket_server, constants.OWN_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received own features")
        
        # receive other clients features
        self.features_other_node = receive_with_header(self.socket_server, constants.OTHER_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received other nodes features")
        
        # receive own test features
        self.test_features = receive_with_header(self.socket_server, constants.TEST_FEATURES)
        logger.info(f"{self.name} received test features")
    
    
    def run_iteration(self, test=False):
        if test:
            features = self.test_features
        else:
            features = self.features
        # receive weights from server
        thetas = receive_with_header(self.socket_server, constants.THETAS, expected_dtype=np.ndarray)
        results = []
        for i, feature in enumerate(features):
            logger.debug(f"{self.name} Running feature number {i}")
            single_result = self.run_circuit_locally(feature, thetas)
            results.append(single_result)
        
        send_with_header(self.socket_server, results, constants.RESULTS)

    
    def run_circuit_locally(self, feature: float, weights: list[float]):
        
        n_shots = self.params["n_shots"]
        q_depth = self.params["q_depth"]
        
        results_arr = []
        
        for i in range(n_shots):
            logger.debug(f"{self.name} is executing shot number {i+1} of {n_shots} shots")
            
            # create Qubit and apply future map
            q = Qubit(self.netqasm_connection)
            ry_feature_map(q, feature)
            
            # we can have max. 5 qubits active at once
            # therefore if we want a depth > 4, we need to execute the hidden layer in multiple rounds
            max_eprs = constants.MAX_VALUES["eprs"] # maximum value of EPR pairs that can be generated at once (due to hardware limitations)
            epr_chunks = generate_chunks_with_max_size(max_eprs, q_depth) # get the chunks of epr pairs. depth 10 and max_eprs 4 would yield [4,4,2]
            for j, n_pairs in enumerate(epr_chunks):
                eprs = self.create_or_recv_epr_pairs(n_pairs)
                # execute hidden layer
                for k in range(len(eprs)):
                    logger.debug(f"{self.name} entered layer: {j * max_eprs + k + 1}")
            
                    # apply theta rotation
                    q.rot_Y(angle=weights[j * max_eprs + k])
                    # apply remote CNOT between clients
                    if self.ctrl_qubit:
                        remote_cnot_control(self.socket_client, q, eprs[k])
                    else:
                        remote_cnot_target(self.socket_client, q, eprs[k])
            # apply one more rotation in the end
            q.rot_Y(angle=weights[-1])        
            
            # measure
            result = q.measure()
            
            self.netqasm_connection.flush()
            results_arr.append(result.value)
            
            
        return results_arr
            
    
    def create_or_recv_epr_pairs(self, n_pairs: int):
        if self.ctrl_qubit:
            # create epr pairs
            assert self.socket_client.recv(block=True) == "ACK"
            self.netqasm_connection.flush()
            eprs = self.epr_socket_other_client.create_keep(number=n_pairs)
            logger.debug(f"{self.name} generated {n_pairs} epr pairs")
            
        else:
            # receive epr pairs
            self.socket_client.send("ACK")
            self.netqasm_connection.flush()
            eprs = self.epr_socket_other_client.recv_keep(number=n_pairs)
            logger.debug(f"{self.name} received {n_pairs} epr qubits")
        return eprs
    