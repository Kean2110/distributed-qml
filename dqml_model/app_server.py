import os
import shutil
import utils.constants as constants
from utils.helper_functions import save_classification_report, remove_folder_except, load_latest_input_checkpoint
from utils.logger import setup_logging
from utils.config_parser import ConfigParser
from utils.timer import global_timer
from utils.plotting import plot_data_with_moving_average
import matplotlib.pyplot as plt
import server

def main(app_config=None):
    """
    Instantiate the server and start the training of the DQML model.
    """
    config = ConfigParser(None, None) # reinstantiate in case we start our app with "netqasm simulate" (then main.py is not executed)
    # set output path for current run
    output_path = os.path.join(constants.APP_BASE_PATH, "output", f"{config.run_id}") 
    # setup the output folder and the logging
    setup_output_folder(output_path, config.config_path) 
    setup_logging(config.enable_netqasm_logs, output_path, config.log_level)
    # instantiate server
    server_instance = server.QMLServer(config.n_qubits, config.epochs, config.initial_thetas, config.random_seed, config.q_depth, config.n_shots, config.n_samples, config.test_size, config.dataset_function, config.start_from_checkpoint, output_path)
    fname = f"netqasm_{server_instance.n_shots}shots_{server_instance.q_depth}qdepth_{config.n_samples}samples"
    try:
        # train and test the model
        report = server_instance.train_and_test_model(fname)
        # fill report with execution times
        report["execution_avgs"] = global_timer.get_execution_averages()
        report["execution_times"] = global_timer.get_execution_times()
        save_classification_report(fname, output_path, report)
        # plot execution times
        plot_output_path = os.path.join(output_path, "plots")
        plot_execution_times(plot_output_path, report["execution_times"])
        
    except Exception as e:
        print("An error occured in server: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
        
    return {
        "report": report,
        "thetas": server_instance.thetas
    }
    

def main_test_only():
    """
    Only test the DQML Model. Inputs need to be provided in the "inputs/" folder, and the "input_run_name" needs to be adjusted.
    """
    # load latest checkpoint from input weights folder
    input_run_name = "iris_4shots_4depth_randomseed49"
    output_run_name = f"TEST_ONLY_{input_run_name}"
    input_run_dir = os.path.join(constants.APP_BASE_PATH, "inputs", input_run_name)
    input_checkpoint_dir = os.path.join(input_run_dir, "checkpoints")
    data = load_latest_input_checkpoint(input_checkpoint_dir)
    if "config" in data: # if we have a saved config in the checkpoint (e.g. from qiskit runs)
        config = data["config"]
    else: # load config manually from yaml
        config_path = os.path.join(input_run_dir, "config.yaml")
        config = ConfigParser(config_path, output_run_name)
    # set output path
    output_path = os.path.join(constants.APP_BASE_PATH, "output", output_run_name) 
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # setup logging
    setup_logging(False, output_path, "DEBUG")
    # instantiate server with trained theta values
    thetas = data["params"]
    server_instance = server.QMLServer(config.n_qubits, config.epochs, thetas, config.random_seed, config.q_depth, config.n_shots, config.n_samples, config.test_size, config.dataset_function, False, output_path)
    # send params and features to clients
    server_instance.send_params_and_features()
    # execute test
    report = server_instance.test_model()
    fname = f"test_only_report.txt"
    save_classification_report(fname, output_path, report)
    
    return {
        "report": report,
        "thetas": server_instance.thetas
    }
    

def setup_output_folder(output_folder_path: str, config_path: str):
    """
    Create output folder and move config into it.
    
    :param output_folder_path: Path of the folder.
    :param config_path: Path of the config.
    """
    # if output folder exists
    if os.path.exists(output_folder_path):
        # delete all contents except checkpoints folder
        remove_folder_except(output_folder_path, ["checkpoints"])
    else:
        os.mkdir(output_folder_path)
    # copy config to output folder
    shutil.copy(config_path, os.path.join(output_folder_path, "config.yaml"))
    
    
def plot_execution_times(plot_output_path: str, execution_times: dict):
    """
    Generate execution time plots from the saved execution times.
    :param plot_output_path: Path of directory where plots are stored.
    :param execution_times: Dict with execution_times
    """
    for key, value in execution_times.items():
        if len(value) > 1: # only plot if function was executed more than once
            filename = key + "_times.png" # names of the plots are the names of the timed functions
            window_size = 10
            plot_data_with_moving_average(filename, plot_output_path, value, window_size)
    
    
if __name__ == "__main__":
    # entry point for netqasm
    main()
        



