import os
import yaml
import uuid

class ConfigParser:
    _instance = None
    num_iter = 100
    enable_netqasm_logging = False
    batch_size = 16
    learning_rate = 0.01
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
            setattr(self, key, value)
    
    def get_config(self) -> dict:
        return self.config