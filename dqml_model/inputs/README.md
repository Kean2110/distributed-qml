# Instructions for test only run
1. Put trained files in this folder. Put the required files (the checkpoint directory with the checkpoint files + the config.yaml) in a directory with the input run name
2. Adjust the paths in the "main_test_only" function in the app_server.py
3. Go to the main file, and make sure that the create_app() function is called with the argument test_only set to True
4. Execute python main.py
5. The test output will be generated in the regular output folder with the prefix "TEST_ONLY_"