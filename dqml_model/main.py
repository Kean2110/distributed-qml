import warnings
from netqasm.runtime.application import default_app_instance
from netqasm.runtime.interface.config import default_network_config, NetworkConfig, Node, Link, Qubit, NoiseType
from netqasm.sdk.external import simulate_application
from utils.config_parser import ConfigParser
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
    """
    Sets up the config based on the provided command line arguemtns.
    If two arguments are provied, the first one is the config number, and the second one is the RUN ID.
    If one argument is provided, the RUN ID is ommitted.
    If no argument is provided, the default config is used.
    """
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


def create_network_config(noise_type: NoiseType = NoiseType.NoNoise, required_qubits_per_qpu: int = 5, fidelity=1) -> NetworkConfig:
    """
    Creates a custom Network Config including Noise and default amounts of Qubits per QPU.
    
    :param noise_type: Type of Quantum Channel noise.
    :param required_qubits_per_qpu: Number of qubits that each QPU will have.
    :param fidelity: Quantum link fidelity.
    
    :returns: NetworkConfig.
    """
    
    links = []
    
    # configure all three nodes
    # server doesnt contain qubits, clients each contain "required_qubits_per_qpu" qubits
    node_server = Node(name=utils.constants.NODE_NAMES[0], hardware=utils.constants.DEFAULT_HW, qubits=[], gate_fidelity=1)
    node_client1 = Node(name=utils.constants.NODE_NAMES[1], hardware=utils.constants.DEFAULT_HW, qubits = [Qubit(id=i, t1=0, t2=0) for i in range(required_qubits_per_qpu)], gate_fidelity=1)
    node_client2 = Node(name=utils.constants.NODE_NAMES[2], hardware=utils.constants.DEFAULT_HW, qubits = [Qubit(id=i, t1=0, t2=0) for i in range(required_qubits_per_qpu)], gate_fidelity=1)
    
    nodes = [node_server, node_client1, node_client2]
    
    # setup links between all network nodes
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
                fidelity=fidelity,
            )
            links += [link]
    
    network_config = NetworkConfig(nodes, links)
    return network_config


def create_app(test_only=False):
    """
    Creates the NetQASM Application and simulates it.
    
    :param test_only: True if a test only run should be executed.
    """
    # set server_main function
    if test_only:
        server_main = app_server.main_test_only
    else:
        server_main = app_server.main
    
    # create a NetQASM App Instance
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
            # setup Network Config
            if c.use_default_network_config:
                network_config = default_network_config(utils.constants.NODE_NAMES, utils.constants.DEFAULT_HW)
            else:
                # configure network config based on the provided config parameters (required EPR pairs and data qubits)
                required_qubits_per_qpu = len(c.layers_with_rcnot) + int(c.n_qubits / 2)
                c.max_qubits_per_qpu = min(required_qubits_per_qpu, utils.constants.MAX_VALUES["qubits_per_client"])
                network_config = create_network_config(c.noise_model, c.max_qubits_per_qpu, c.link_fidelity)
        
        # simulate the netqasm application
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
    # change to True if we want to have a test only run, modify paths in "server.py", and copy all data into inputs dir
    create_app(False)