from ctypes import util
import warnings
import os
from netqasm.runtime.application import default_app_instance
from netqasm.runtime.interface.config import default_network_config, NetworkConfig, Node, Link, QuantumHardware, Qubit, NoiseType
from netqasm.sdk.external import simulate_application
from utils.config_parser import ConfigParser
import glob
import yaml
import logging
import os
import sys
import traceback
import utils.constants
import app_server
import app_client1
import app_client2

"""
Entry point if we don't want to use `netqasm simulate`, e.g. for debugging
"""   

def setup_config():
    try:
        config_number = int(sys.argv[1])
        config_path = f"config/config{config_number}.yaml"
        try:
            run_id_str = sys.argv[2]
            c = ConfigParser(config_path, run_id_str)
        except IndexError:
            warnings.warn("No run ID provided, using randomly generated run ID")  
            c = ConfigParser(config_path, None)
    except IndexError:
        warnings.warn("No config ID provided, using default 'config.yaml' config")
        config_path = "config.yaml"
        c = ConfigParser(config_path, None)
    return c
        

def read_params_from_yaml():
    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Search for YAML files in the current directory
    yaml_files = glob.glob(os.path.join(current_directory, '*.yaml'))
    inputs = {}
    # read yaml files and add to inputs
    for file in yaml_files:
        with open(file, 'r') as config_file:
            data = yaml.safe_load(config_file) or {}
            # obtain file_name of yaml by removing directory path and file extension (.yaml)
            file_name = os.path.basename(file)
            instance_name, _ = os.path.splitext(file_name)
            inputs[instance_name] = data
    return inputs


def create_network_config(noise_type: NoiseType = NoiseType.NoNoise, required_qubits_per_qpu: int = 5) -> NetworkConfig:
    links = []
    
    node_server = Node(name=utils.constants.NODE_NAMES[0], hardware=utils.constants.DEFAULT_HW, qubits=[], gate_fidelity=1)
    node_client1 = Node(name=utils.constants.NODE_NAMES[1], hardware=utils.constants.DEFAULT_HW, qubits = [Qubit(id=i, t1=0, t2=0) for i in range(required_qubits_per_qpu)], gate_fidelity=1)
    node_client2 = Node(name=utils.constants.NODE_NAMES[2], hardware=utils.constants.DEFAULT_HW, qubits = [Qubit(id=i, t1=0, t2=0) for i in range(required_qubits_per_qpu)], gate_fidelity=1)
    
    nodes = [node_server, node_client1, node_client2]
    
    for node in nodes:
        node_name = node.name
        for other_node in nodes:
            other_node_name = other_node.name
            if other_node_name == node_name:
                continue
            link = Link(
                name=f"link_{node_name}_{other_node_name}",
                node_name1=node_name,
                node_name2=other_node_name,
                noise_type=noise_type,
                fidelity=1,
            )
            links += [link]
    
    network_config = NetworkConfig(nodes, links)
    return network_config


def create_app(test_only=False):
    if test_only:
        server_main = app_server.main_test_only
    else:
        server_main = app_server.main
        
    app_instance = default_app_instance(
        [
            ("server", server_main),
            ("client1", app_client1.main),
            ("client2", app_client2.main)
        ]
    )
    
    try:
        if test_only:
            network_config = default_network_config(utils.constants.NODE_NAMES, utils.constants.DEFAULT_HW)
        else:
            c = setup_config()
            if c.use_default_network_config:
                network_config = default_network_config(utils.constants.NODE_NAMES, utils.constants.DEFAULT_HW)
            else:
                required_qubits_per_qpu = len(c.layers_with_rcnot) + int(c.n_qubits / 2)
                c.max_qubits_per_qpu = required_qubits_per_qpu
                network_config = create_network_config(c.noise_model, required_qubits_per_qpu)
            
        simulate_application(
            app_instance,
            use_app_config=False,
            post_function=None,
            enable_logging=False,
            network_cfg=network_config
        )
    except:
        traceback.print_exc()
        sys.exit(1)
    
    
if __name__ == "__main__":
    create_app(False)