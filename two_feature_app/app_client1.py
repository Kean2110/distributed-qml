import client
    
def main(app_config=None):
    try:
        client1 = client.Client("client1", "client2", 0, 2, 0, True)
        client1.start_client()
    except Exception as e:
        print("An error occured in client 1: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()
 

if __name__ == "__main__":
    main()
