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
        self.iter = None
        self.file_name = f"checkpoint_{time.strftime('%Y%m%dT%H%M%S')}.pickle"
        # set and create output dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
    
    def save_intermediate_results(self, params, loss, iter):
        if self.best_loss is not None and loss >= self.best_loss:
            pass
        else:
            self.best_loss = loss
            self.best_params = params
            self.iter = iter
            file_location = os.path.join(self.checkpoint_dir, self.file_name)
            intermediate_dict = {'params': self.best_params, 'loss': self.best_loss, 'iter': self.iter}
            logger.info("Saving intermediate results to pickle file.")
            with open(file_location, 'wb') as file:   
                pickle.dump(intermediate_dict, file)