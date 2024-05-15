import os
import shutil
from utils.helper_functions import save_classification_report
from utils.logger import setup_logging
from utils.config_parser import ConfigParser
import server

def main(app_config=None):
    config = ConfigParser(None, None)
    output_path = setup_output_folder(f"output/{config.config_id}/", config.config_path)
    setup_logging(config.enable_netqasm_logging, output_path)
    server_instance = server.QMLServer(config.max_iter, config.initial_thetas, config.random_seed, config.q_depth, config.n_shots, config.n_samples, config.test_size, config.dataset_function, config.start_from_checkpoint)
    fname = f"netqasm_{server_instance.n_shots}shots_{server_instance.q_depth}qdepth_{config.n_samples}samples"
    try: 
        report = server_instance.run_gradient_free(fname, output_path)
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


def setup_output_folder(folder_path, config_path):
    # Create output folder and copy config into it
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    shutil.copy(config_path, folder_path + "config.yaml")
    return os.path.abspath(folder_path)
    
    
if __name__ == "__main__":
    main()
        



