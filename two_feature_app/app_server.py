from utils.helper_functions import save_classification_report
from utils.logger import setup_logging
import server

def main(app_config=None, enable_netqasm_logging=False, num_iter=1, initial_thetas=None, batch_size=1, learning_rate=0.01, random_seed=42, q_depth=1, n_shots=1, n_samples=100, test_size=0.2, dataset_function="iris"):
    setup_logging(enable_netqasm_logging)
    server_instance = server.QMLServer(num_iter, initial_thetas, batch_size, learning_rate, random_seed, q_depth, n_shots, n_samples, test_size, dataset_function)
    fname = f"netqasm_{server_instance.n_shots}shots_{server_instance.q_depth}qdepth_{n_samples}samples.png"
    plot_name = fname + ".png"
    report_name = fname + ".txt"
    try: 
        report = server_instance.run_gradient_free(plot_name)
        report = server_instance.test_gradient_free()
        save_classification_report(report, report_name)
    except Exception as e:
        print("An error occured in server: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
        
    return {
        "report": report,
        "thetas": server_instance.thetas
    }

    
if __name__ == "__main__":
    main()
        



