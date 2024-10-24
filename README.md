# distributed-qml
Distributed Quantum Machine Learning Master's Thesis

# Setup and Installation
1. Make sure you have a python version 3.9.x installed
2. Create a virtual environment or conda environment
3. Activate the environment
4. Install NetQASM: pip install netqasm
5. Install SquidASM \
    4.1. Install NetSquid: https://netsquid.org/#registration \
    4.2. Install SquidASM: pip install squidasm --extra-index-url=https://{netsquid-user-name}:{netsquid-password}@pypi.netsquid.org
6. pip install -r requirements.txt
- To create a pip requirements.txt use pip list --format=freeze > requirements.txt
- Watch out that you install the NetQASM and Squidasm versions that are specified in the requirements.txt

# Repo structure

This repository consists of different applications, which are each in a sub-folder. The main ones are:

## Two Feature App

The NetQASM App of the distributed quantum machine learning algorithm
-main.py - Entry point of the application, if not run with 'netqasm simulate'.
-app_ files - The NetQASM apps, two clients and a server
-client.py - The Client class, instantiated by app_client1 and app_client2
-server.py - The Server class, instantiated by app_server
-config.yaml files - Default config for the project. If no configs are provided in as console args or the app is run with "netqasm simulate", this config is used
-config/ - Contains the configs
-utils/ - Contains util files, like for plotting, logging, parsing the config files, etc
-output/ - The output where the runs outputs are stored, such as the log, the plots, the checkpoints, a copy of the config


## Two Feature Qiskit
The Qiskit implementation and baseline of the NetQASM implementation
- main.py entry point for the run
- helper_functions.py functions for e.g. plotting, calculating the parity, etc
- config.py the run config
- two_feature_qiskit.ipynb - the VQC implementation as a notebook (independent of the one defined in main.py)
- vqc_benchmark - the notebook as a python file
- plots/ - the output dir for the plots
- classification_reports/ - the output dir for the classification reports

# Running the NetQASM application


## Locally

### Run with netqasm

Go to the directory of the netqasm application (cd two_feature_app) and run "netqasm simulate".
This will use the config.yaml as a config file.
By specifying the log-level as "--log-level=DEBUG" u can include the netqasm DEBUG logs in your run.
The outputs of the run can be found in the folder output/$RUN_ID/, where RUN_ID is a randomly generated ID.

### Run manually (recommended)
Navigate to the netqasm application (cd two_feature_app) and run "python main.py [$config_number] [$run_id]
Both Run ID and Config Number can be ommitted. In that case the default config.yaml config will be used, and a run_id conforming to the UUID standard will be assigned.

Before executing make sure your config, is stored in the config/ folder with the name "config[$config_number].yaml".
An example config file can be found in the config.yaml.

The outputs of the run will be stored in outputs/[$run_id].

# Running on SLURM
To run on slurm, prepare all config files you want to run in the folder config/ with the names config[$config_number].yaml. Those number have to be added in the job.sh file as the --array parameter.
You can make further adaptions, e.g. to get notified by mail.
Then navigate to the directory of the job.sh and execute "sbatch job.sh".

# Running using remote SSH machine
Since executions take a lot of time to run, you can keep it running on a remote SSH server without being connected to the server all the time.
For that u can use screen.
1. Connect to remote server via SSH (ssh user@server)
2. clone repo and do setup in this README
3. execute the command "screen", this will start a new process
4. start your run (see sections "run manually" or "run with NetQASM")
5. press Ctrl + A and Ctrl + D to detach your session, but keep the process running
6. return to your session anytime by pressing executing "screen -r"

# Additional notes
- Since the runtimes can be long, make sure that you do not override the config file that was used for a currently used run. In case the run is stopped and attempts to re-run, the same config file will be used.