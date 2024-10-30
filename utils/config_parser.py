import os
import yaml
import uuid
import threading

from utils import constants

class ConfigParser:
    """
    Singleton class ConfigParser.
    When being reinstantiated, it returns the current config.
    """
    _instance = None
    _lock = threading.Lock()
    epochs = 100
    enable_netqasm_logs = False
    random_seed = 42
    q_depth = 4
    n_shots = 32
    n_samples = 100
    test_size = 0.2
    initial_thetas = None
    # either MOONS or IRIS
    dataset_function = "MOONS"
    start_from_checkpoint = False
    n_qubits = 2
    layers_with_rcnot = list(range(q_depth)) # per default all layers have a RCNOT

    def __new__(cls, config_path=None, run_id=None):
        if cls._instance is None:
            with cls._lock:
                cls._instance = super().__new__(cls)
                cls._instance.load_config(config_path)
                cls._instance.set_id(run_id)
        return cls._instance

    def set_id(self, run_id):
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = str(uuid.uuid1())

    def load_config(self, config_path):
        # if no config path provided, load the default config
        if not config_path:
            config_path = "config.yaml"
        app_dir = os.getcwd()
        config_file = os.path.join(app_dir, config_path)
        # load config
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config_path = config_file
        for key, value in self.config.items():
            ConfigParser.check_restrictions(key, value)
            setattr(self, key, value)
        ConfigParser.check_epr_list_restriction(self.layers_with_rcnot, self.q_depth)
    
    def get_config(self) -> dict:
        return self.config
    
    @staticmethod
    def check_restrictions(key, value):
        if key in constants.MIN_VALUES:
            min = constants.MIN_VALUES[key]
            if value < min:
                raise ValueError(f"The provided value for {key} is too small (must be at least {min})")
        if key in constants.MAX_VALUES:
            max = constants.MAX_VALUES[key]
            if value > max:
                raise ValueError(f"The provided value for {key} is too big. It must be <= {max}") 
            
    @staticmethod
    def check_epr_list_restriction(list_of_epr_layers: list, q_depth: int):
        if max(list_of_epr_layers) >= q_depth:
            raise ValueError(f"Layer {max(list_of_epr_layers)} in the list of layers with RCNOTs must be smaller than the depth {q_depth} of the circuit")
        if min(list_of_epr_layers) < 0:
            raise ValueError(f"Layers must be positive in the list of RCNOT layers.")
        if len(list_of_epr_layers) > q_depth:
            raise ValueError(f"RCNOT layer list {list_of_epr_layers} is too long for depth {q_depth}")
        if len(list_of_epr_layers) != len(set(list_of_epr_layers)):
            raise ValueError(f"RCNOT layer list {list_of_epr_layers} contains duplicates.")
