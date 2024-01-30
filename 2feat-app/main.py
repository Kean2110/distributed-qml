from netqasm.runtime.application import default_app_instance
from netqasm.sdk.external import simulate_application
from netqasm.runtime.debug import run_application
import glob
import yaml
import os
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
        


def create_app():
    app_instance = default_app_instance(
        [
            ("server", app_server.main),
            ("client1", app_client1.main),
            ("client2", app_client2.main)
        ]
    )
    
    app_instance.program_inputs = read_params_from_yaml()
    
    simulate_application(
        app_instance,
        use_app_config=False,
        post_function=None,
        enable_logging=False
    )
    
if __name__ == "__main__":
    create_app()