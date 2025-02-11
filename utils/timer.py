import functools
import time
from utils.logger import logger

import numpy as np

"""
Class Timer that contains a decorator, that can be applied to measure function executions.
Saves all execution times and averages in respective dictionaries.
https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
"""
class Timer:
    def __init__(self) -> None:
        self.execution_times = {}
        self.execution_avg = {}
    
    def timer(self, func):
        func_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(run_time)
            return value

        return wrapper 
    
    
    def calculate_averages(self):
        for key, value in self.execution_times.items():
            # save as list to save it with pickle dump
            self.execution_avg[key] = np.mean(value).tolist()
            
    
    def get_execution_time(self, key):
        try:
            return self.execution_times[key]
        except KeyError:
            logger.warning(f"Cannot return execution time for key {key}")
            return [-1.0]
    
    def get_execution_times(self):
        return self.execution_times
    
    
    def set_execution_times(self, times: dict):
        self.execution_times = times
        
    
    def get_execution_average(self, key):
        try:
            return self.execution_avg[key]
        except KeyError:
            logger.warning(f"Cannot return execution time for key {key}")
            return -1.0
    
    
    def get_execution_averages(self):
        if not self.execution_avg:
            self.calculate_averages()
        return self.execution_avg


global_timer = Timer()