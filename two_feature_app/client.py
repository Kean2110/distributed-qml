import math

import numpy as np
import utils.constants as constants
from logging import Logger
from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from netqasm.sdk import EPRSocket, Qubit
from utils.helper_functions import phase_gate
from utils.socket_communication import receive_with_header, send_as_str
from netqasm.logging.glob import get_netqasm_logger
from utils.qubit_communication import remote_cnot_control, remote_cnot_target

logger = get_netqasm_logger()


class Client:
    def __init__(self, name: str, other_client_name: str, socket_id_with_server: int, socket_id_with_other_client: int, epr_socket_id_with_server: int, logger: Logger, ctrl_qubit: bool):
        self.name = name
        self.socket_server = Socket(name, "server", socket_id=socket_id_with_server)
        self.socket_client = Socket(name, other_client_name, socket_id=socket_id_with_other_client)
        self.epr_socket_server = EPRSocket(remote_app_name="server", epr_socket_id=0, remote_epr_socket_id=epr_socket_id_with_server)
        self.epr_socket_other_client = EPRSocket(remote_app_name=other_client_name, epr_socket_id=1, remote_epr_socket_id=1)
        self.ctrl_qubit = ctrl_qubit
        self.eprs_needed_for_feature_map = 2
        self.params = None
        self.features = None
        self.features_other_node = None
        
        self.netqasm_connection = NetQASMConnection(
            app_name=name,
            epr_sockets=[self.epr_socket_server, self.epr_socket_other_client],
            max_qubits=10
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
                else:
                    raise ValueError("Unregistered instruction received")
    

    def receive_starting_values_from_server(self):
        # receive number of iters
        self.params = receive_with_header(self.socket_server, constants.PARAMS)
        logger.info(f"{self.name} received params dict: {self.params}")
        
        self.features = receive_with_header(self.socket_server, constants.OWN_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received own features")
        
        self.features_other_node = receive_with_header(self.socket_server, constants.OTHER_FEATURES, expected_dtype=np.ndarray)
        logger.info(f"{self.name} received other nodes features")
    
    
    def run_iteration(self):
        # receive thetas from server
        thetas = receive_with_header(self.socket_server, constants.THETAS, expected_dtype=np.ndarray)
        
        results = []
        for i, feature in enumerate(self.features):
            single_result = self.run_circuit_locally(feature, thetas, self.features_other_node[i])
            results.append(single_result)
        
        send_as_str(self.socket_server, results)

    
    def run_circuit_locally(self, feature: float, weights: list[float], feature_other_node: float):
       
        n_shots = self.params["n_shots"]
        q_depth = self.params["q_depth"]
        
        results_dict = {"0": 0, "1": 0}
        
        for i in range(n_shots):
            logger.info(f"{self.name} is executing shot number {i+1} of {n_shots} shots")
            
            eprs = self.create_or_recv_epr_pairs(q_depth + self.eprs_needed_for_feature_map)
            
            q = Qubit(self.netqasm_connection)
            
            self.rx_feature_map(q, feature)
            #if self.ctrl_qubit:
            #    self.zz_feature_map_ctrl(q, eprs[:self.eprs_needed_for_feature_map], feature)
            #else:
            #    self.zz_feature_map_target(q, eprs[:self.eprs_needed_for_feature_map], feature, feature_other_node)
            
            eprs = eprs[self.eprs_needed_for_feature_map:]
            
            # execute hidden layer
            for j in range(q_depth):
                logger.info(f"{self.name} entered layer: {j+1}")
                
                # apply theta rotation
                q.rot_Y(angle=weights[j])
                print(f"{self.name}: {eprs[j]}")
                if self.ctrl_qubit:
                    remote_cnot_control(self.socket_client, self.netqasm_connection, q, eprs[j])
                else:
                    remote_cnot_target(self.socket_client, self.netqasm_connection, q, eprs[j])
                    
            # measure
            result = q.measure()
            self.netqasm_connection.flush()
            
            results_dict[str(result)] += 1
            
        return max(results_dict, key = lambda x: results_dict[x])
            
    
    def create_or_recv_epr_pairs(self, n_pairs: int):
        if self.ctrl_qubit:
            # create epr pairs
            eprs = self.epr_socket_other_client.create_keep(number=n_pairs)
            logger.info(f"{self.name} generated {n_pairs} epr pairs")
        else:
            # receive epr pairs
            eprs = self.epr_socket_other_client.recv_keep(number=n_pairs)
            logger.info(f"{self.name} received {n_pairs} epr pairs")
        return eprs

    
    def zz_feature_map_ctrl(self, qubit: Qubit, eprs: list[Qubit], feature: float):
        qubit.H()
        phase_gate(2 * feature, qubit)
        remote_cnot_control(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        remote_cnot_control(self.socket_client, self.netqasm_connection, qubit, eprs[1])
        logger.info(f"{self.name} executed zz feature map control")
        
    
    def zz_feature_map_target(self, qubit: Qubit, eprs: list[Qubit], feature: float, feature_other_node: list[float]):
        qubit.H()
        phase_gate(angle = 2 * feature, qubit = qubit)
        remote_cnot_target(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        phase_gate(angle = 2 * (math.pi - feature_other_node) * (math.pi - feature), qubit = qubit)
        remote_cnot_target(self.socket_client, self.netqasm_connection, qubit, eprs[0])
        logger.info(f"{self.name} executed zz feature map target")
        
        
    def rx_feature_map(self, qubit: Qubit, feature: float):
        qubit.rot_X(angle=feature)