import logging
from netqasm.logging.glob import get_netqasm_logger
import os
import sys

logger = logging.getLogger("DQML")

def setup_logging(enable_netqasm_logs: bool, output_path: str, log_level: str):
    """
    Setup logging configuration
    
    :param enable_netqasm_logs: Whether to include logs produced by NetQASM.
    :param output_path: Logging output directory.
    :param log_level: Common log levels (NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL)
    """
    # return netqasm logger if we want netqasm logs, else our own
    if enable_netqasm_logs:
        logger = get_netqasm_logger()
    else:
        logger = logging.getLogger("DQML")
    # set up path
    log_path = os.path.join(output_path, f"{log_level}.log")
    logger.setLevel(log_level)
    # Files have log level of maximum of DEBUG, whereas the console has a maximum level of INFO.
    consoleFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fileFormatter = logging.Formatter("%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s")
    fileHandler = logging.FileHandler(log_path, mode="w")
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)