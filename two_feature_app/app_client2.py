import client

def main(app_config=None):
    try:
        client2 = client.Client("client2", "client1", 1, 2, 1, False)
        client2.start_client()
    except Exception as e:
        print("An error occured in client 2: ", e)
        import traceback, sys
        traceback.print_exc()
        sys.exit()

      
if __name__ == "__main__":
    main()
