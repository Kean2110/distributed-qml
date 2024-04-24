from utils.logger import setup_logging
import server

def main(app_config=None, enable_netqasm_logging=False, num_iter=1, initial_thetas=None, batch_size=1, learning_rate=0.01, random_seed=42, q_depth=1, n_shots=1, n_samples=100, dataset_function="iris"):
    setup_logging(enable_netqasm_logging)
    server_instance = server.QMLServer(num_iter, initial_thetas, batch_size, learning_rate, random_seed, q_depth, n_shots, n_samples, dataset_function)
    try: 
        server_instance.run_gradient_free("batch_loss_gradient_free.png")
    except Exception as e:
        print("An error occured in server: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
        
    return {
        "results": server_instance.all_results.tolist(),
        "thetas": server_instance.thetas
    }

    
if __name__ == "__main__":
    main()
        



