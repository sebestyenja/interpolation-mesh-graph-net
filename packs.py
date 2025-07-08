import torch

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=False)
class RawDataPack:
    cells : torch.tensor
    mesh_pos : torch.tensor
    target : torch.tensor
    node_type : torch.tensor
    
    def move_to_device(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            setattr(self, field, value.to(device))
            
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)

@dataclass(frozen=False)
class NodePack:
    node_features : torch.tensor
    simulated_indices : torch.tensor
    scripted_indices : torch.tensor
    fixed_indices : torch.tensor
    
    def move_to_device(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            setattr(self, field, value.to(device))
            
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)

@dataclass(frozen=False)
class EdgePack:

    edge_features : torch.tensor
    edge_idx : torch.tensor
        
    coarse_edge_features : Any
    coarse_edge_idx : Any
        
    aggr_edge_weights : Any
    aggr_edge_idx : Any
    
    int_edge_weights : Any
    int_edge_receivers : Any
    int_edge_senders : Any
    
    def move_to_device(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            setattr(self, field, value.to(device))
            
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
            
@dataclass(frozen=False)
class TargetPack:
    normalized_target : torch.tensor
    target_velocity : torch.tensor
    
    def move_to_device(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            setattr(self, field, value.to(device))
            
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
            
@dataclass(frozen=False)
class GraphDataPack:
    node_pack : dataclass
    edge_pack : dataclass
    target_pack : dataclass
    
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
        
@dataclass(frozen=False)
class TrainingParameterPack:
    
    dataset_name : str
    n_epochs : int
    latent_size : int
    norm_info_accumulation_steps : int
    message_passes : int
    
    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
    
@dataclass(frozen=False)
class TestingParameterPack:
    
    monitor_interval : int
    test_interval : int
    test_sequence_index : int

    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
            
@dataclass(frozen=False)
class OptimizerParameterPack:
    
    lr : float
    lr_gamma : float
    norm_acc_length: int
    grad_limit : float

    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)
        
@dataclass(frozen=False)
class NetworkParameterPack:
    
    name : str
    layer_count : int
    in_size : int
    hidden_size : int
    out_size : int
    norm_in : bool
    norm_out : bool
    multi_mlp_count : int
    device : str

    def __iter__(self):
        return (getattr(self, field) for field in self.__dataclass_fields__)