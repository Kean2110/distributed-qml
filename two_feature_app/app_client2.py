import client
from utils import constants
from utils.config_parser import ConfigParser

def main(app_config=None):
    try:
        client2 = client.Client("client2", "client1", constants.SOCKET_SERVER_C2, constants.SOCKET_C1_C2, False)
        client2.start_client()
    except Exception as e:
        print("An error occured in client 2: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()

      
if __name__ == "__main__":
    main()
