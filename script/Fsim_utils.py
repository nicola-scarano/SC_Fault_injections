
from pytorchfi.FI_Weights import FI_manager 
from pytorchfi.FI_Weights import DatasetSampling 

from torch.utils.data import DataLoader, Subset
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Sampler
from collections import Counter
import numpy as np
import random

class StratifiedBatchSampler(object):
    def __init__(self, dataset, num_images:int, random_state:int=None) -> None:
        self.dataset = dataset
        # num images per class
        self.num_images=num_images
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def make_weights_for_balanced_classes(self):                                                                         
        count = Counter(self.dataset.classes)                                                   
        weights = list()                          
        N = float(sum(count)) 
        for elem in self.dataset:
            # counter for the single class devided by the total number of elements    
            weights.append(count[elem[1]]/N)                                        
        return weights
    
    def get_indices(self, nclasses:int):
        weights = self.make_weights_for_balanced_classes()
        all_indices = np.arange(0,len(self.dataset))
        indices = np.random.choice(a=all_indices, size=nclasses, p=weights)
        return indices
    
    def __len__(self, ):
        return len(self.indices)



def setup_test_dataloader(config, test_data_loader):
    test_batch_size=config['test']['test_data_loader']['batch_size']
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    subsampler = StratifiedBatchSampler(test_data_loader.dataset,5)
    index_dataset=subsampler.get_indices()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)
    return dataloader

