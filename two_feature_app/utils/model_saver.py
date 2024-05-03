import json
import os
import pickle
from utils.logger import logger

import numpy as np


class ModelSaver:
    def __init__(self) -> None:
        self.best_loss = None
        self.best_params = None
    
    def save_intermediate_results(self, filename, params, loss):
        if self.best_loss is not None and loss >= self.best_loss:
            pass
        else:
            self.best_loss = loss
            self.best_params = params
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
            param_dir = os.path.join(parent_dir, "parameters/")
            file_location = os.path.join(param_dir, filename + ".pickle")
            intermediate_dict = {'params': self.best_params, 'loss': self.best_loss}
            logger.info("Saving intermediate results to pickle file.")
            with open(file_location, 'wb') as file:   
                pickle.dump(intermediate_dict, file)