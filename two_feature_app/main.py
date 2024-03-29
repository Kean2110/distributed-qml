from netqasm.runtime.application import default_app_instance
from netqasm.sdk.external import simulate_application
from netqasm.runtime.debug import run_application
from netqasm.logging.glob import get_netqasm_logger, set_log_level
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
        

def setup_logging():
    curr_dir = os.path.dirname(os.path.relpath(__file__))
    log_path = os.path.join(curr_dir, "log/output.log")
    logger = get_netqasm_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileHandler = logging.FileHandler(log_path, mode="w")
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)


def create_app():
    app_instance = default_app_instance(
        [
            ("server", app_server.main),
            ("client1", app_client1.main),
            ("client2", app_client2.main)
        ]
    )
    
    app_instance.program_inputs = read_params_from_yaml()
    
    try:
        setup_logging()
        
        simulate_application(
            app_instance,
            use_app_config=False,
            post_function=None,
            enable_logging=True
        )
    except:
        traceback.print_exc()
    
if __name__ == "__main__":
    create_app()