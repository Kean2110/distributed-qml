import os
import yaml
import uuid

from utils import constants

class ConfigParser:
    """
    Singleton class ConfigParser.
    When being reinstantiated, it returns the current config.
    """
    _instance = None
    max_iter = 100
    enable_netqasm_logging = False
    random_seed = 42
    q_depth = 4
    n_shots = 32
    n_samples = 100
    test_size = 0.2
    initial_thetas = None
    # either MOONS or IRIS
    dataset_function = "MOONS"
    start_from_checkpoint = False

    def __new__(cls, config_path=None, config_id=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config(config_path)
            cls._instance.set_id(config_id)
        return cls._instance

    def set_id(self, config_id):
        if config_id:
            self.config_id = config_id
        else:
            self.config_id = str(uuid.uuid1())

    def load_config(self, config_path):
        # if no config path provided, load the default config
        if not config_path:
            config_path = "config.yaml"
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        config_file = os.path.join(parent_dir, config_path)
        # load config
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config_path = config_file
        for key, value in self.config.items():
            ConfigParser.check_restrictions(key, value)
            setattr(self, key, value)
    
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