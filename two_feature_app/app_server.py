from netqasm.logging.glob import get_netqasm_logger
import server

logger = get_netqasm_logger()


def main(app_config=None, num_iter=1, initial_thetas=None, batch_size=1, learning_rate=0.01, random_seed=42, q_depth=1, n_shots=1, dataset_function="iris"):
    server_instance = server.QMLServer(num_iter, initial_thetas, batch_size, learning_rate, random_seed, q_depth, n_shots, dataset_function)
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
        



