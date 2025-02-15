import client
from utils import constants
from utils.config_parser import ConfigParser
    
def main(app_config=None):
    try:
        client1 = client.Client("client1", "client2", constants.SOCKET_SERVER_C1, constants.SOCKET_C1_C2, True)
        client1.start_client()
    except Exception as e:
        print("An error occured in client 1: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()


if __name__ == "__main__":
    main()
