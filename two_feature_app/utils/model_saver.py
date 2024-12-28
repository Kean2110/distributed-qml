import json
import os
import pickle
import time
from utils.constants import LOSS_TOL, MIN_SAVE_INTERVAL
from utils.logger import logger

import numpy as np


class ModelSaver:
    def __init__(self, output_dir, best_loss=None) -> None:
        self.best_loss = best_loss
        self.best_params = None
        self.iter = 0
        # set and create output dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
    
    def save_intermediate_results(self, params, iter, losses, accs, bypass_loss_check=False):
        loss = losses[-1] # current loss is the last loss in the list
        # make an extra save if no save in the last MIN_SAVE_INTERVAL iters and loss is better than LOSS_TOL
        make_extra_save = iter >= self.iter + MIN_SAVE_INTERVAL and loss <= LOSS_TOL
        # save either if its's first loss, best loss, bypassed, or an extra save
        if self.best_loss is None or loss <= self.best_loss or bypass_loss_check or make_extra_save:
            file_name = f"checkpoint_{time.strftime('%Y%m%dT%H%M%S')}.pickle"
            self.best_loss = loss
            self.best_params = params
            self.iter = iter
            file_location = os.path.join(self.checkpoint_dir, file_name)
            intermediate_dict = {'params': self.best_params, 'iter': self.iter, 'losses': losses, 'accs': accs}
            logger.info("Saving intermediate results to pickle file.")
            with open(file_location, 'wb') as file:   
                pickle.dump(intermediate_dict, file)