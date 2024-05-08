import json
import os
import pickle
import time
from utils.logger import logger

import numpy as np


class ModelSaver:
    def __init__(self, output_dir) -> None:
        self.best_loss = None
        self.best_params = None
        self.file_name = f"checkpoint_{time.strftime('%Y%m%dT%H%M%S')}.pickle"
        # set and create output dir
        self.output_dir = os.path.join(output_dir, "parameters/")
        os.mkdir(self.output_dir)
    
    def save_intermediate_results(self, params, loss):
        if self.best_loss is not None and loss >= self.best_loss:
            pass
        else:
            self.best_loss = loss
            self.best_params = params
            file_location = os.path.join(self.output_dir, self.file_name)
            intermediate_dict = {'params': self.best_params, 'loss': self.best_loss}
            logger.info("Saving intermediate results to pickle file.")
            with open(file_location, 'wb') as file:   
                pickle.dump(intermediate_dict, file)