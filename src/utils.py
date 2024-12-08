import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pynvml
import random
import heapq

from datetime import datetime

from typing import Iterable

def str_formatted_datetime():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def select_best_device(mode):
    if mode not in ["m", "u"]:
        raise Exception(f'select_device_with_most_free_memory: Acceptable inputs for mode are "m" (most free memory) and "u" (least utilization_). You specified: {mode}')
    
    indices = list(range(torch.cuda.device_count()))
    random.shuffle(indices) #shuffle the indices we iterate through so that, if, say, a bunch of processes scramble for GPUs at once, the first one won't get them all

    if mode == "m":
        max_free_memory = 0
        device_index = 0
        for i in indices:
            torch.cuda.set_device(i)
            free_memory = torch.cuda.mem_get_info()[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                device_index = i
        return torch.device(f"cuda:{device_index}")

    elif mode == "u":
        pynvml.nvmlInit()
        min_util = 100
        device_index = 0
        for i in indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # Get the handle for the target GPU
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu  # GPU utilization percentage (integer)
            if gpu_utilization < min_util:
                min_util = gpu_utilization
                device_index = i
        pynvml.nvmlShutdown()

        # If all the GPUs are basically at max util, then make choice via memory availiability
        if min_util > 95:
            return select_best_device(mode="m")

        return torch.device(f"cuda:{device_index}")



class Min_n_Items:

    def __init__(self, n:int):
        '''
        Keeps a running record of the n lowest values so far and their associated epoch number
        Values are something like loss values, etc
        '''

        self.n = n
        self.heap = []
    
    def record_epoch(self, val:float, epoch_num:int):
        '''
        Records a value with associated epoch number
        If the value is smaller than the existing n values already in the heap, it replaces the largest value in the heap with the new value
        Else, the value is not recorded
        '''

        #when given tuple, heapq sorts using first element. We negate value because heapq is a min-heap
        epoch_tuple = (-val, epoch_num)
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, epoch_tuple)
        else:
            heapq.heappushpop(self.heap, epoch_tuple)
    
    def get_smallest_val(self, include_epoch=False):
        '''
        Returns the smallest recorded value

        include_epoch (bool): Include epoch number of smallest value too
        '''
        
        if include_epoch:
            return -max(self.heap)[0], max(self.heap)[1]

        return -max(self.heap)[0]

    def get_dict(self):
        '''
        Returns a dict with the recorded values and their associated epoch values
        '''
        return {epoch_tuple[1]:-epoch_tuple[0] for epoch_tuple in self.heap}

    def __str__(self):
        '''
        Prints out a string with information about recorded values
        As you can see it is currently tailored for validation loss
        '''
        min_str = f"Top {self.n} Epochs Info:\n"
        min_str += f"Average Val Loss Across Lowest {self.n} epochs: {self.avg_min_n_val()}\n"
        min_n_dict = self.get_dict()
        sorted_items_by_value = sorted(min_n_dict.items(), key=lambda item: item[1])
        for key, value in sorted_items_by_value:
            min_str += f"\tEpoch: {key}\tVal Loss: {value}\n"
        return min_str

    def avg_min_n_val(self, n=None):
        '''
        Return the average value across the n smallest values
        If n is None then uses self.n (all tracked values)
        '''
        if n is None:
            n = self.n
        elif n > self.n:
            warings.warn(f"Min_n_Items average_min_n_val: Specified value n cannot be greater than tracked number of items. You specified: {n}. Defaulting to {self.n}, the number of all tracked items.")
            n = self.n
        min_dict = self.get_dict()
        sorted_min_vals = sorted(min_dict.values())
        return sum(sorted_min_vals[:n])/n



# taken from myself from code originally written for my HTLMTL project (while visiting Gomes Group @ CMU)
class MTL_DataLoader():

    '''
    An iterable. Given a tuple of dataloaders, will return a tuple containing a batch from each one every time it is called. When a dataloader is out of batches, this class will simple reset the dataloader and keep on returning batches.
    
    '''

    def __init__(self, dataloaders:tuple[DataLoader], batches_per_epoch:int):
        self.dataloaders = dataloaders
        self.batches_per_epoch = batches_per_epoch

        self.iterables = [iter(dataloader) for dataloader in self.dataloaders]
    
    def __iter__(self):
        self.batches_left = self.batches_per_epoch
        return self

    def __next__(self):
        if self.batches_left > 0:
            self.batches_left -= 1
            batches = []
            for i in range(len(self.iterables)):
                iterable = self.iterables[i]
                try:
                    batches.append(next(iterable))
                except StopIteration:
                    iterable = iter(self.dataloaders[i])
                    self.iterables[i] = iterable
                    batches.append(next(iterable))
            return tuple(batches)
        else:
            raise StopIteration




class Running_Avg():
    '''
    Maintains a running average of values added
    '''
    def __init__(self):
        self.running_avg = 0.0
        self.counter = 0
    
    def add(self, val:float):
        self.running_avg = (self.running_avg * self.counter + val)/(self.counter + 1)
        self.counter += 1

    def get_avg(self):
        return self.running_avg


class Multi_Running_Avgs():
    '''
    Maintains separate Running_Avg objects at once
    '''
    def __init__(self, n:int):
        self.n = n
        self.running_avgs = [Running_Avg() for _ in range(n)]

    def add(self, vals:Iterable[float]):
        if len(vals) != self.n:
            raise ValueError(f"Multi_Running_Avgs.add(): Length of iterable given ({len(vals)}) does not match expected length ({self.n})")
        
        [running_avg.add(val) for running_avg, val in zip(self.running_avgs, vals)]

    def get_running_avgs(self)->list[float]:
        '''
        Return a list containing n floats representing each of the running averages
        ''' 

        return [running_avg.get_avg() for running_avg in self.running_avgs]

    def get_summed_avgs(self)->float:
        '''
        Returns the sum of all running averages
        '''

        return sum(self.get_running_avgs())