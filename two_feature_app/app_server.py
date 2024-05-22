import os
import shutil
import utils.constants as constants
from utils.helper_functions import save_classification_report, remove_folder_except
from utils.logger import setup_logging
from utils.config_parser import ConfigParser
from utils.timer import global_timer
import server

def main(app_config=None):
    config = ConfigParser(None, None)
    output_path = os.path.join(constants.PROJECT_BASE_PATH, "output", f"{config.config_id}")
    setup_output_folder(output_path, config.config_path)
    setup_logging(config.enable_netqasm_logging, output_path)
    server_instance = server.QMLServer(config.max_iter, config.initial_thetas, config.random_seed, config.q_depth, config.n_shots, config.n_samples, config.test_size, config.dataset_function, config.start_from_checkpoint, output_path)
    fname = f"netqasm_{server_instance.n_shots}shots_{server_instance.q_depth}qdepth_{config.n_samples}samples"
    try:
        report = server_instance.run_gradient_free(fname)
        global_timer.calculate_averages()
        report["execution_times"] = global_timer.get_execution_times()
        report["execution_avgs"] = global_timer.get_execution_averages()
        save_classification_report(fname, output_path, report)
    except Exception as e:
        print("An error occured in server: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
        
    return {
        "report": report,
        "thetas": server_instance.thetas
    }


def setup_output_folder(output_folder_path: str, config_path: str):
    # Create output folder and copy config into it
    # if output folder exists
    if os.path.exists(output_folder_path):
        # delete all contents except checkpoints folder
        remove_folder_except(output_folder_path, ["checkpoints"])
    else:
        os.mkdir(output_folder_path)
    shutil.copy(config_path, os.path.join(output_folder_path, "config.yaml"))
    
    
if __name__ == "__main__":
    main()
        



