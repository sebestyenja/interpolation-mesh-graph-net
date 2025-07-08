import torch
import torch.nn as nn
import numpy as np

class cumulative_normalizer(nn.Module):
    def __init__(self, name, size, max_accumulations=10**6, std_epsilon=1e-6, saving_path='./', device='cuda:0'):

        super(cumulative_normalizer, self).__init__()
        self._name = name
        self.size = size
        self.device = device

        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=self.device)
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=self.device)[None, None]
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=self.device)[None, None]
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=self.device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=self.device)

        self.saving_path = saving_path
        
        self.freeze = False
       
    def forward(self, batch, accumulate):
        if accumulate == True and self.freeze == False:
            if self._num_accumulations < self._max_accumulations:
                if batch.shape[0] > 0:
                    self._accumulate_stats(batch.detach())      
        batch = (batch - self._mean().to(batch.device)) / self._std_with_epsilon().to(batch.device) 
        return batch

    def inverse(self, normalized_batch_data):
        return normalized_batch_data * self._std_with_epsilon().to(normalized_batch_data.device) + self._mean().to(normalized_batch_data.device)
    
    def _accumulate_stats(self, batch):
        count = batch.shape[0]
        data_sum = torch.sum(batch, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batch**2, axis=0, keepdims=True)
        
        self._acc_sum = self._acc_sum + data_sum
        self._acc_sum_squared = self._acc_sum_squared + squared_data_sum
        self._acc_count = self._acc_count + count
        self._num_accumulations = self._num_accumulations + 1
                
    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count
    
    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
        std = torch.nan_to_num(torch.maximum(std, self._std_epsilon), self._std_epsilon)
        return std