import json
import os
import pickle
import time
from utils.logger import logger

import numpy as np


class ModelSaver:
    def __init__(self, output_dir, best_loss=None) -> None:
        self.best_loss = best_loss
        self.best_params = None
        self.iter = None
        # set and create output dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
    
    def save_intermediate_results(self, params, iter, losses, accs, exec_times, bypass_loss_check=False):
        loss = losses[-1] # current loss is the last loss in the list
        if self.best_loss is None or loss < self.best_loss or bypass_loss_check:
            file_name = f"checkpoint_{time.strftime('%Y%m%dT%H%M%S')}.pickle"
            self.best_loss = loss
            self.best_params = params
            self.iter = iter
            file_location = os.path.join(self.checkpoint_dir, file_name)
            intermediate_dict = {'params': self.best_params, 'iter': self.iter, 'losses': losses, 'accs': accs, 'exec_times': exec_times}
            logger.info("Saving intermediate results to pickle file.")
            with open(file_location, 'wb') as file:
                pickle.dump(intermediate_dict, file, protocol=pickle.HIGHEST_PROTOCOL)