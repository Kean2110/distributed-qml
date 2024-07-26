import warnings
import os
os.environ["NETQASM_SIMULATOR"] = "netsquid"
from netqasm.runtime.application import default_app_instance
from netqasm.sdk.external import simulate_application
from netqasm.runtime.debug import run_application
from netqasm.logging.glob import get_netqasm_logger, set_log_level
from utils.logger import setup_logging
from utils.config_parser import ConfigParser
import glob
import yaml
import logging
import os
import sys
import traceback
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
        if not test_only:
            setup_config()
        
        simulate_application(
            app_instance,
            use_app_config=False,
            post_function=None,
            enable_logging=False
        )
    except:
        traceback.print_exc()
        sys.exit(1)
    
    
if __name__ == "__main__":
    create_app(False)