import logging
from netqasm.logging.glob import get_netqasm_logger
import os
import sys

logger = logging.getLogger("DQML")

def setup_logging(enable_netqasm_logs, output_path):
    if enable_netqasm_logs:
        logger = get_netqasm_logger()
    else:
        logger = logging.getLogger("DQML")
    log_path = os.path.join(output_path, "debug.log")
    logger.setLevel(logging.DEBUG)
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