import numpy as np
import pickle
import torch

from packs import NodePack
from packs import EdgePack
from packs import TargetPack
from packs import GraphDataPack
from packs import RawDataPack

from utils import get_aggr_weights
from utils import get_int_weights
from utils import build_node_graph_edges_from_triangle_cells
from utils import make_rotation_matrix
from utils import rotate_positions
from utils import build_rel_pos_features

class static_dataset():
    def __init__(self, ds_path, training_folder, training_noise, mode, element_order, normalizer_pack, device):
        
        self.device = device
        
        self.ds_path = ds_path
        
        self.training_folder = training_folder
        
        self.noise_level = training_noise
        
        self.mode_flag = mode

        self.element_order = element_order[0]
        
        self.node_normalizer = normalizer_pack[0]
        self.edge_normalizer = normalizer_pack[1]
        self.target_normalizer = normalizer_pack[2]
        self.coarse_edge_normalizer = normalizer_pack[3]
        
        self.item_names = ["cells", "mesh_pos", "target", "node_type"]
        
    def _set_iterator(self, dataset):
        index_list = np.arange(len(dataset))[:, None]
        zeros = np.zeros_like(index_list)
        index_list = np.concatenate((index_list, zeros), axis=1)
        return index_list
        
    def load_dataset(self, ds_path, test_it=None):
        self.dataset = self._get_arrays(ds_path, test_it)
        self.iterator = self._set_iterator(self.dataset)
        print("Loading dataset is complete!") if test_it is None else print("Loading testset is complete!")
              
    def _get_set_of_order(self, loaded_data, order):
        data = []
        for d in loaded_data:
            cells = torch.from_numpy(d[order + "-" + self.item_names[0]]).to(device=self.device).long()
            mesh_pos = torch.from_numpy(d[order + "-" + self.item_names[1]]).to(device=self.device)
            target = torch.from_numpy(d[order + "-" + self.item_names[2]]).to(device=self.device)
            node_type = torch.from_numpy(d[order + "-" + self.item_names[3]]).to(device=self.device).long()
            data.append([cells, mesh_pos, target, node_type])
        return data
        
    def _get_arrays(self, ds_path, it=None):
        with open(ds_path, "rb") as f:
            records = pickle.load(f) 
        records = self._get_set_of_order(records, self.element_order)
        if it is not None:
            records = [records[it]]            
        return records
    
    def __len__(self):
        return self.iterator.shape[0]
    
    def shuffle(self):
        np.random.shuffle(self.iterator)   

    def get_data(self, index, rotate_flag=True):
        with torch.no_grad():
            raw_data = self._get_raw_data(index)
            return self._build_graph_data(raw_data, rotate_flag)

    def reverse_output(self, network_output, data_pack):
        network_output_signs = torch.where(network_output >= 0.0, 1, -1)
        normalized_prediction = (torch.pow(np.e * torch.ones(network_output.shape), network_output.abs()) - 1) * network_output_signs     
        prediction = self.target_normalizer.inverse(normalized_prediction)
        return data_pack.mesh_pos, data_pack.target, prediction

    def _get_raw_data(self, index):
        
        file_index = self.iterator[index, 0]
        
        data_pack = self.dataset[file_index]
        
        cells = data_pack[0].to(self.device)
        mesh_pos = data_pack[1].to(self.device)
        target = data_pack[2].to(self.device)
        node_type = data_pack[3].to(self.device)

        if self.noise_level > 1e-8:
            disturbed_nodes = node_type[:, [0, 3]].sum(dim=1).nonzero().view(-1)
            noise_field = torch.randn(mesh_pos.shape).to(mesh_pos.device) * self.noise_level
            mesh_pos[disturbed_nodes] += noise_field[disturbed_nodes]

        raw_data_pack = RawDataPack(cells, mesh_pos, target, node_type)
                  
        return raw_data_pack

    def _build_graph_data(self, raw_data_pack, rotate_flag):
        
        cells = raw_data_pack.cells.long()
        mesh_pos = raw_data_pack.mesh_pos.float()
        target = raw_data_pack.target.float()
        node_type = raw_data_pack.node_type.float()

        all_node_indices = torch.arange(mesh_pos.shape[0]).long()
        self.original_node_indices = all_node_indices
        
        non_wall_node_indices = node_type[:, 0].nonzero().view(-1)
        self.non_wall_node_indices = non_wall_node_indices
        
        fixed_node_indices = node_type[:, 1].nonzero().view(-1)
        self.fixed_node_indices = fixed_node_indices
        
        load_node_indices = node_type[:, 2].nonzero().view(-1)
        self.load_node_indices = load_node_indices
        
        wall_node_indices = node_type[:, 3].nonzero().view(-1)
        self.wall_node_indices = wall_node_indices

        load_direction = torch.tensor((0.0, 1.0, 0.0)).to(mesh_pos.device)

        displacemenets = target[:, :2]
        stress = target[:, 2]
                              
        cells = torch.cat((torch.arange(cells.shape[0])[:, None], cells), dim=1)
        
        order = int(self.element_order) if (self.element_order.isdigit()) else 1 # follow order if mesh mode - first order if interpolation or multiscale
        order = 1 if self.element_order == "hq" else order # hq mesh is always first order
        
        receivers, senders = build_node_graph_edges_from_triangle_cells(cells, order) # mgn graph
         
        coarse_nodes = cells[:, 1:4].unique() # coarse graph
        all_nodes = cells[:, 1:].unique() # full ho graph
            
        if self.mode_flag == "INTERPOLATION" or self.mode_flag == "MULTISCALE":
            order = int(self.element_order) if self.element_order.isdigit() else input("Element order error in - dataset._build_graph_data()!")
    
            coarse_receivers, coarse_senders = build_node_graph_edges_from_triangle_cells(cells, 1)

            # weights for aggregating the fine graph into the coarse graph
            # receivers = coarse nodes
            # senders = fine nodes
            aggr_weights, aggr_receivers, aggr_senders = get_aggr_weights(mesh_pos[all_nodes], cells[:, 1:], mesh_pos[coarse_nodes], cells[:, 1:4], receivers, senders)        
            
            # weights for interpolating the coarse graph onto the fine graph
            # receivers = fine nodes
            # senders = coarse nodes
            int_weights, int_receivers, int_senders = get_int_weights(mesh_pos[all_nodes], cells[:, 1:], mesh_pos[coarse_nodes], cells[:, 1:4])
        else:
            aggr_weights, aggr_receivers, aggr_senders, int_weights, int_receivers, int_senders = None, None, None, None, None, None
            
        if rotate_flag == True:
            ### support data for rotation
            z_rot_angle = torch.rand((1)) * 2 * np.pi
            z_rot_matrix = make_rotation_matrix(mesh_pos, z_rot_angle, "z")
            
            mesh_pos = torch.cat((mesh_pos, torch.zeros((mesh_pos.shape[0], 1)).to(mesh_pos.device)), dim=1)
            mesh_pos = rotate_positions(mesh_pos, z_rot_matrix)[:, :2]

            displacemenets = torch.cat((displacemenets, torch.zeros((displacemenets.shape[0], 1)).to(displacemenets.device)), dim=1)
            displacemenets = rotate_positions(displacemenets, z_rot_matrix)[:, :2]

            load_direction = torch.mm(z_rot_matrix[0], load_direction[:, None]).squeeze()[:2]
        else:
            load_direction = load_direction[:2]

        target = torch.cat((displacemenets, stress[:, None]), dim=1)

        normalized_target = self.target_normalizer.forward(target, accumulate=True).squeeze(dim=0)
    
        edge_features = build_rel_pos_features([mesh_pos], receivers, senders)
        edge_features = self.edge_normalizer.forward(edge_features, accumulate=True).detach().clone()
                
        node_features = torch.cat((load_direction[None, :].expand(node_type.shape[0], -1), node_type), dim=1).detach().clone()
        node_features = self.node_normalizer.forward(node_features, accumulate=True)
        
        node_pack = NodePack(node_features, all_node_indices, torch.empty((0)).long(), torch.empty((0)).long())
                    
        if self.mode_flag == "INTERPOLATION" or self.mode_flag == "MULTISCALE":
            coarse_edge_features = build_rel_pos_features([mesh_pos], coarse_receivers, coarse_senders)
            coarse_edge_features = self.coarse_edge_normalizer.forward(coarse_edge_features, accumulate=True).detach().clone()

            coarse_edge_idx = torch.cat((coarse_receivers[:, None], coarse_senders[:, None]), dim=1)
            aggr_edge_idx = torch.cat((aggr_receivers[:, None], aggr_senders[:, None]), dim=1)
                        
            edge_pack = EdgePack(edge_features, torch.cat((receivers[:, None], senders[:, None]), dim=1),
                                 coarse_edge_features, coarse_edge_idx,
                                 aggr_weights, aggr_edge_idx,
                                 int_weights, int_receivers, int_senders)
        else:
            edge_pack = EdgePack(edge_features, torch.cat((receivers[:, None], senders[:, None]), dim=1),
                                 None, None, None, None, None, None, None)

        target_pack = TargetPack(normalized_target, target)
                        
        graph_data_pack = GraphDataPack(node_pack, edge_pack, target_pack)

        return graph_data_pack