from netqasm.logging.glob import get_netqasm_logger
import client

logger = get_netqasm_logger()
    
def main(app_config=None):
    try:
        client1 = client.Client("client1", "client2", 0, 2, 0, logger, True)
        client1.start_client()
    except Exception as e:
        print("An error occured in client 1: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
 

if __name__ == "__main__":
    main()
