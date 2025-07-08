import torch
from torch_scatter import scatter_mean

def build_node_graph_edges_from_triangle_cells(cells, order=1, sort=True):
    if order == 1 or order == "1":
        # first order triangle elements
        cells = cells[:, 1:]
        
        edge_1 = cells[:, [0, 1]]
        edge_2 = cells[:, [1, 2]]
        edge_3 = cells[:, [2, 0]]

        edges = torch.cat((edge_1, edge_2, edge_3), dim=0)
            
    elif order == 2 or order == "2":
        # second order triangle elements
        cells = cells[:, 1:]
        
        edge_1 = cells[:, [0, 4]]
        edge_2 = cells[:, [4, 2]]
        edge_3 = cells[:, [3, 5]]

        edge_4 = cells[:, [0, 3]]
        edge_5 = cells[:, [3, 1]]
        edge_6 = cells[:, [4, 5]]

        edge_7 = cells[:, [1, 5]]
        edge_8 = cells[:, [5, 2]]
        edge_9 = cells[:, [3, 4]]

        edges = torch.cat((edge_1, edge_2, edge_3,
                            edge_4, edge_5, edge_6, 
                            edge_7, edge_8, edge_9), dim=0)

    elif order == 3 or order == "3":
        # third order triangle elements
        cells = cells[:, 1:]
        
        edge_1 = cells[:, [0, 5]]
        edge_2 = cells[:, [5, 6]]
        edge_3 = cells[:, [6, 2]]
        edge_4 = cells[:, [3, 9]]
        edge_5 = cells[:, [9, 8]]
        edge_6 = cells[:, [4, 7]]

        edge_7 = cells[:, [0, 3]]
        edge_8 = cells[:, [3, 4]]
        edge_9 = cells[:, [4, 1]]
        edge_10 = cells[:, [5, 9]]
        edge_11 = cells[:, [9, 7]]
        edge_12 = cells[:, [6, 8]]
        
        edge_13 = cells[:, [1, 7]]
        edge_14 = cells[:, [7, 8]]
        edge_15 = cells[:, [8, 2]]
        edge_16 = cells[:, [4, 9]]
        edge_17 = cells[:, [9, 6]]
        edge_18 = cells[:, [3, 5]]

        edges = torch.cat((edge_1, edge_2, edge_3, edge_4, edge_5, edge_6,
                           edge_7, edge_8, edge_9, edge_10, edge_11, edge_12,
                           edge_13, edge_14, edge_15, edge_16, edge_17, edge_18), dim=0)
    
    elif order == 4 or order == "4":
        # fourth order triangle elements
        cells = cells[:, 1:]
        
        edge_1 = cells[:, [0, 6]]
        edge_2 = cells[:, [6, 7]]
        edge_3 = cells[:, [7, 8]]
        edge_4 = cells[:, [8, 2]]
        edge_5 = cells[:, [3, 12]]
        edge_6 = cells[:, [12, 13]]
        edge_7 = cells[:, [13, 11]]
        edge_8 = cells[:, [4, 14]]
        edge_9 = cells[:, [14, 10]]
        edge_10 = cells[:, [5, 9]]
        
        edge_11 = cells[:, [0, 3]]
        edge_12 = cells[:, [3, 4]]
        edge_13 = cells[:, [4, 5]]
        edge_14 = cells[:, [5, 1]]
        edge_15 = cells[:, [6, 12]]
        edge_16 = cells[:, [12, 14]]
        edge_17 = cells[:, [14, 9]]
        edge_18 = cells[:, [7, 13]]
        edge_19 = cells[:, [13, 10]]
        edge_20 = cells[:, [8, 11]]
        
        edge_21 = cells[:, [1, 9]]
        edge_22 = cells[:, [9, 10]]
        edge_23 = cells[:, [10, 11]]
        edge_24 = cells[:, [11, 2]]
        edge_25 = cells[:, [5, 14]]
        edge_26 = cells[:, [14, 13]]
        edge_27 = cells[:, [13, 8]]
        edge_28 = cells[:, [4, 12]]
        edge_29 = cells[:, [12, 7]]
        edge_30 = cells[:, [3, 6]]

        edges = torch.cat((edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7, edge_8, edge_9, edge_10,
                           edge_11, edge_12, edge_13, edge_14, edge_15, edge_16, edge_17, edge_18, edge_19, edge_20, 
                           edge_21, edge_22, edge_23, edge_24, edge_25, edge_26, edge_27, edge_28, edge_29, edge_30), dim=0)
        
    elif order == 5 or order == "5":
        # fifth order triangle elements
        cells = cells[:, 1:]
        
        edge_1 = cells[:, [0, 7]]
        edge_2 = cells[:, [7, 8]]
        edge_3 = cells[:, [8, 9]]
        edge_4 = cells[:, [9, 10]]
        edge_5 = cells[:, [10, 2]]
        edge_6 = cells[:, [3, 15]]
        edge_7 = cells[:, [15, 16]]
        edge_8 = cells[:, [16, 17]]
        edge_9 = cells[:, [17, 14]]
        edge_10 = cells[:, [4, 18]]
        edge_11 = cells[:, [18, 19]]
        edge_12 = cells[:, [19, 13]]
        edge_13 = cells[:, [5, 20]]
        edge_14 = cells[:, [20, 12]]
        edge_15 = cells[:, [6, 11]]
        
        edge_16 = cells[:, [0, 3]]
        edge_17 = cells[:, [3, 4]]
        edge_18 = cells[:, [4, 5]]
        edge_19 = cells[:, [5, 6]]
        edge_20 = cells[:, [6, 1]]
        edge_21 = cells[:, [7, 15]]
        edge_22 = cells[:, [15, 18]]
        edge_23 = cells[:, [18, 20]]
        edge_24 = cells[:, [20, 11]]
        edge_25 = cells[:, [8, 16]]
        edge_26 = cells[:, [16, 19]]
        edge_27 = cells[:, [19, 12]]
        edge_28 = cells[:, [9, 17]]
        edge_29 = cells[:, [17, 13]]
        edge_30 = cells[:, [10, 14]]
        
        edge_31 = cells[:, [1, 11]]
        edge_32 = cells[:, [11, 12]]
        edge_33 = cells[:, [12, 13]]
        edge_34 = cells[:, [13, 14]]
        edge_35 = cells[:, [14, 2]]
        edge_36 = cells[:, [6, 20]]
        edge_37 = cells[:, [20, 19]]
        edge_38 = cells[:, [19, 17]]
        edge_39 = cells[:, [17, 10]]
        edge_40 = cells[:, [5, 18]]
        edge_41 = cells[:, [18, 16]]
        edge_42 = cells[:, [16, 9]]
        edge_43 = cells[:, [4, 15]]
        edge_44 = cells[:, [15, 8]]
        edge_45 = cells[:, [3, 7]]

        edges = torch.cat((edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7, edge_8, edge_9, edge_10, edge_11, edge_12, edge_13, edge_14, edge_15,
                           edge_16, edge_17, edge_18, edge_19, edge_20, edge_21, edge_22, edge_23, edge_24, edge_25, edge_26, edge_27, edge_28, edge_29, edge_30,
                           edge_31, edge_32, edge_33, edge_34, edge_35, edge_36, edge_37, edge_38, edge_39, edge_40, edge_41, edge_42, edge_43, edge_44, edge_45), dim=0)
               
    edges = torch.cat((edges, edges.roll(1, 1)), dim=0)
    
    if sort == True:
        edges = edges.unique(dim=0)
        
    return edges[:, 0], edges[:, 1]

def make_rotation_matrix(tensor, rot_angle, direction):
    if direction == "x":            
        rot_matrix = torch.tensor([[1, 0, 0],
                                   [0, torch.cos(rot_angle), - torch.sin(rot_angle)],
                                   [0, torch.sin(rot_angle), torch.cos(rot_angle)]]).float()
    elif direction == "y":            
        rot_matrix = torch.tensor([[torch.cos(rot_angle), 0, torch.sin(rot_angle)],
                                  [0, 1, 0],
                                  [- torch.sin(rot_angle), 0, torch.cos(rot_angle)]]).float()
    elif direction == "z":            
        rot_matrix = torch.tensor([[torch.cos(rot_angle), - torch.sin(rot_angle), 0],
                                  [torch.sin(rot_angle), torch.cos(rot_angle), 0],
                                  [0, 0, 1]]).float()   
    exp_rot_matrix = rot_matrix[None, :, :].expand(tensor.shape[0], -1, -1)

    return exp_rot_matrix 

def make_angle_wise_rotation_matricies(rot_angles, direction):
    if direction == "x":       
        rot_matrix_0 = torch.vstack([torch.ones_like(rot_angles), torch.zeros_like(rot_angles), torch.zeros_like(rot_angles)]).t()
        rot_matrix_1 = torch.vstack([torch.zeros_like(rot_angles), torch.cos(rot_angles), - torch.sin(rot_angles)]).t()
        rot_matrix_2 = torch.vstack([torch.zeros_like(rot_angles), torch.sin(rot_angles), torch.cos(rot_angles)]).t()
    elif direction == "y":   
        rot_matrix_0 = torch.vstack([torch.cos(rot_angles), torch.zeros_like(rot_angles), torch.sin(rot_angles)]).t()
        rot_matrix_1 = torch.vstack([torch.zeros_like(rot_angles), torch.ones_like(rot_angles), torch.zeros_like(rot_angles)]).t()
        rot_matrix_2 = torch.vstack([- torch.sin(rot_angles), torch.zeros_like(rot_angles), torch.cos(rot_angles)]).t()
    elif direction == "z":  
        rot_matrix_0 = torch.vstack([torch.cos(rot_angles), - torch.sin(rot_angles), torch.zeros_like(rot_angles)]).t()
        rot_matrix_1 = torch.vstack([torch.sin(rot_angles), torch.cos(rot_angles), torch.zeros_like(rot_angles)]).t()
        rot_matrix_2 = torch.vstack([torch.zeros_like(rot_angles), torch.zeros_like(rot_angles), torch.ones_like(rot_angles)]).t()
    
    rot_matrix = torch.cat((rot_matrix_0[:, None, :], rot_matrix_1[:, None, :], rot_matrix_2[:, None, :]), dim=1)
    
    return rot_matrix

def rotate_positions(tensor, rot_matrix):
    rotated_tensor = torch.bmm(rot_matrix, tensor[:, :, None]).squeeze(dim=-1)
    return rotated_tensor.reshape((rotated_tensor.shape[0], rotated_tensor.shape[1]))

def build_rel_pos_features(pos_tensor_list: list, receivers, senders, unit_length=False):
    features = []
    for pos in pos_tensor_list:
        rel_pos = pos[senders] - pos[receivers]
        rel_pos_norms = torch.norm(rel_pos, dim=1)[:, None]
        if unit_length == True:
            rel_pos = torch.nn.functional.normalize(rel_pos, dim=1)
        features.append(rel_pos)
        features.append(rel_pos_norms)
    return torch.hstack(features)

def avg_scatter(input_data, receivers, out_shape, detach=True):
    if detach == True:
        out = torch.zeros(out_shape).detach().to(input_data.device)
    else:
        out = torch.zeros(out_shape).to(input_data.device)
    return scatter_mean(input_data, receivers, out=out, dim=0)

def upscale_by_linear_weights_2D(us, weights, receivers, senders, output_node_count=None):

    new_us = (us[senders] * weights[:, :, :, None])
   
    new_us = new_us.sum(dim=-2)
    
    if output_node_count is None:
        output_node_count = receivers.max() + 1
   
    new_us = avg_scatter(new_us.flatten(end_dim=1), receivers.flatten(), (output_node_count, us.shape[1]), detach=False)
        
    return new_us

def get_aggr_weights(new_pos, new_elementwise_indices, pos, elementwise_indices, receiver_nodes, sender_nodes):
    
    new_unique_mesh_nodes = elementwise_indices.unique()
    
    ext_new_unique_mesh_nodes = new_unique_mesh_nodes[:, None, None].expand(-1, elementwise_indices.shape[0], elementwise_indices.shape[1])
    
    ext_elementwise_indices = elementwise_indices[None, :, :].expand(ext_new_unique_mesh_nodes.shape[0], -1, -1)
    
    mask = (ext_new_unique_mesh_nodes == ext_elementwise_indices).any(dim=-1).nonzero()
    
    new_unique_mesh_nodes = mask[:, 0]
    containing_cell_indices = mask[:, 1]
    
    nodewise_containing_cells = elementwise_indices[containing_cell_indices]

    exp_nodewise_containing_cells = nodewise_containing_cells[:, None, :].expand(-1, new_elementwise_indices.shape[1], -1).flatten(end_dim=1)
    
    nodewise_new_containing_cells = new_elementwise_indices[containing_cell_indices]
  
    flat_nodewise_new_containing_cells = nodewise_new_containing_cells.flatten()

    nodewise_new_unique_mesh_nodes = new_unique_mesh_nodes[:, None].expand(-1, nodewise_new_containing_cells.shape[1])

    us = torch.zeros((nodewise_new_containing_cells.shape[0], pos.shape[0])).to(new_pos.device)
    us[torch.arange(us.shape[0]).to(new_pos.device), new_unique_mesh_nodes] = 1
    us = us[:, None, :].expand(-1, new_elementwise_indices.shape[1], -1).flatten(end_dim=1)
  
    x = pos[nodewise_containing_cells][:, :, 0].detach()
    y = pos[nodewise_containing_cells][:, :, 1].detach()
    
    x1 = x[:, 0, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)
    x2 = x[:, 1, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)
    x3 = x[:, 2, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)

    y1 = y[:, 0, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)
    y2 = y[:, 1, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)
    y3 = y[:, 2, None].expand(-1, new_elementwise_indices.shape[1]).flatten(end_dim=1)

    new_x = new_pos[flat_nodewise_new_containing_cells, 0].detach()
    new_y = new_pos[flat_nodewise_new_containing_cells, 1].detach()

    J1 = torch.cat((torch.ones(x1.shape[0], 1, device=new_pos.device), x1[:, None], y1[:, None]), dim=1)
    J2 = torch.cat((torch.ones(x2.shape[0], 1, device=new_pos.device), x2[:, None], y2[:, None]), dim=1)
    J3 = torch.cat((torch.ones(x3.shape[0], 1, device=new_pos.device), x3[:, None], y3[:, None]), dim=1)

    Jacobian = torch.cat((J1[:, None, :], J2[:, None, :], J3[:, None, :]), dim=1)

    reciprocaldetJ = torch.reciprocal(torch.linalg.det(Jacobian))
 
    filler = torch.arange(us.shape[0], device=new_pos.device)

    fi1 = torch.cat((us[filler, exp_nodewise_containing_cells[:, 0], None],
                     us[filler, exp_nodewise_containing_cells[:, 1], None],
                     us[filler, exp_nodewise_containing_cells[:, 2], None]), dim=1)[:, 0]
                    
    fi2 = torch.cat((us[filler, exp_nodewise_containing_cells[:, 0], None],
                     us[filler, exp_nodewise_containing_cells[:, 1], None],
                     us[filler, exp_nodewise_containing_cells[:, 2], None]), dim=1)[:, 1]
                    
    fi3 = torch.cat((us[filler, exp_nodewise_containing_cells[:, 0], None],
                     us[filler, exp_nodewise_containing_cells[:, 1], None],
                     us[filler, exp_nodewise_containing_cells[:, 2], None]), dim=1)[:, 2]

    a1 = torch.cat((fi1[:, None], x1[:, None], y1[:, None]), dim=1)
    a2 = torch.cat((fi2[:, None], x2[:, None], y2[:, None]), dim=1)
    a3 = torch.cat((fi3[:, None], x3[:, None], y3[:, None]), dim=1)

    a = torch.cat((a1[:, None, :], a2[:, None, :], a3[:, None, :]), dim=1)
    a = torch.linalg.det(a) * reciprocaldetJ

    filler = torch.ones(x1.shape[0], 1, device=new_pos.device)

    b1 = torch.cat((filler, fi1[:, None], y1[:, None]), dim=1)
    b2 = torch.cat((filler, fi2[:, None], y2[:, None]), dim=1)
    b3 = torch.cat((filler, fi3[:, None], y3[:, None]), dim=1)

    b = torch.cat((b1[:, None, :], b2[:, None, :], b3[:, None, :]), dim=1)
    b = torch.linalg.det(b) * reciprocaldetJ

    c1 = torch.cat((filler, x1[:, None], fi1[:, None]), dim=1)
    c2 = torch.cat((filler, x2[:, None], fi2[:, None]), dim=1)
    c3 = torch.cat((filler, x3[:, None], fi3[:, None]), dim=1)

    c = torch.cat((c1[:, None, :], c2[:, None, :], c3[:, None, :]), dim=1)
    c = torch.linalg.det(c) * reciprocaldetJ

    new_us = a + b * new_x + c * new_y

    new_edges = torch.cat((nodewise_new_unique_mesh_nodes[:, :, None], nodewise_new_containing_cells[:, :, None]), dim=-1).flatten(end_dim=1)
    
    new_us = torch.round(new_us, decimals=3)
    
    zero_mask = (new_us > 0.001).nonzero().squeeze()
 
    new_edges = new_edges[zero_mask]
    new_us = new_us[zero_mask]

    new_edges, return_index = uniqueXT(new_edges, return_index=True, dim=0)

    new_us = new_us[return_index].detach()

    return new_us, new_edges[:, 0].long(), new_edges[:, 1].long()
    
def uniqueXT(x, sorted=True, return_index=False, return_inverse=False, return_counts=False, occur_last=False, dim=None):
    if return_index or (not sorted and dim is not None):
        unique, inverse, counts = torch.unique(x, sorted=True,
            return_inverse=True, return_counts=True, dim=dim)
        inv_sorted, inv_argsort = inverse.flatten().sort(stable=True)

        if occur_last and return_index:
            tot_counts = (inverse.numel() - 1 - 
                torch.cat((counts.new_zeros(1),
                counts.flip(dims=[0]).cumsum(dim=0)))[:-1].flip(dims=[0]))
        else:
            tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        
        index = inv_argsort[tot_counts]
        
        if not sorted:
            index, idx_argsort = index.sort()
            unique = (unique[idx_argsort] if dim is None else
                torch.index_select(unique, dim, idx_argsort))
            if return_inverse:
                idx_tmp = idx_argsort.argsort()
                inverse.flatten().index_put_((inv_argsort,), idx_tmp[inv_sorted])
            if return_counts:
                counts = counts[idx_argsort]

        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_inverse:
            ret += (inverse,)
        if return_counts:
            ret += (counts,)
        return ret if len(ret)>1 else ret[0]
    
    else:
        return torch.unique(x, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

def get_int_weights(new_pos, new_elementwise_indices, pos, elementwise_indices): # keep
    
    ext_new_elementwise_indices = new_elementwise_indices[:, :, None].expand(-1, -1, elementwise_indices.shape[1])

    ext_elementwise_indices = elementwise_indices[:, None, :].expand(-1, new_elementwise_indices.shape[1] * elementwise_indices.shape[1], -1)

    us = torch.zeros((ext_elementwise_indices.shape[0], new_elementwise_indices.shape[1],  elementwise_indices.shape[1], elementwise_indices.shape[1]), device=new_pos.device).detach()
    us[:, :, 0, 0] = 1
    us[:, :, 1, 1] = 1
    us[:, :, 2, 2] = 1
    us = us.flatten(start_dim=1, end_dim=2)

    x = pos[ext_elementwise_indices][:, :, :, 0].detach()
    y = pos[ext_elementwise_indices][:, :, :, 1].detach()

    x1 = x[:, :, 0].flatten()
    x2 = x[:, :, 1].flatten()
    x3 = x[:, :, 2].flatten()

    y1 = y[:, :, 0].flatten()
    y2 = y[:, :, 1].flatten()
    y3 = y[:, :, 2].flatten()

    new_x = new_pos[ext_new_elementwise_indices.flatten(start_dim=1)][:, :, 0].detach().flatten()
    new_y = new_pos[ext_new_elementwise_indices.flatten(start_dim=1)][:, :, 1].detach().flatten()

    J1 = torch.cat((torch.ones(x1.shape[0], 1, device=new_pos.device), x1[:, None], y1[:, None]), dim=1)
    J2 = torch.cat((torch.ones(x2.shape[0], 1, device=new_pos.device), x2[:, None], y2[:, None]), dim=1)
    J3 = torch.cat((torch.ones(x3.shape[0], 1, device=new_pos.device), x3[:, None], y3[:, None]), dim=1)

    Jacobian = torch.cat((J1[:, None, :], J2[:, None, :], J3[:, None, :]), dim=1)

    reciprocaldetJ = torch.reciprocal(torch.linalg.det(Jacobian))

    fi1 = torch.cat((us.flatten(end_dim=1)[:, 0, None],
                     us.flatten(end_dim=1)[:, 1, None],
                     us.flatten(end_dim=1)[:, 2, None]), dim=1)[:, 0]
    
    fi2 = torch.cat((us.flatten(end_dim=1)[:, 0, None],
                     us.flatten(end_dim=1)[:, 1, None],
                     us.flatten(end_dim=1)[:, 2, None]), dim=1)[:, 1]
        
    fi3 = torch.cat((us.flatten(end_dim=1)[:, 0, None],
                     us.flatten(end_dim=1)[:, 1, None],
                     us.flatten(end_dim=1)[:, 2, None]), dim=1)[:, 2]

    a1 = torch.cat((fi1[:, None], x1[:, None], y1[:, None]), dim=1)
    a2 = torch.cat((fi2[:, None], x2[:, None], y2[:, None]), dim=1)
    a3 = torch.cat((fi3[:, None], x3[:, None], y3[:, None]), dim=1)

    a = torch.cat((a1[:, None, :], a2[:, None, :], a3[:, None, :]), dim=1)
    a = torch.linalg.det(a) * reciprocaldetJ

    filler = torch.ones(x1.shape[0], 1, device=new_pos.device)

    b1 = torch.cat((filler, fi1[:, None], y1[:, None]), dim=1)
    b2 = torch.cat((filler, fi2[:, None], y2[:, None]), dim=1)
    b3 = torch.cat((filler, fi3[:, None], y3[:, None]), dim=1)

    b = torch.cat((b1[:, None, :], b2[:, None, :], b3[:, None, :]), dim=1)
    b = torch.linalg.det(b) * reciprocaldetJ

    c1 = torch.cat((filler, x1[:, None], fi1[:, None]), dim=1)
    c2 = torch.cat((filler, x2[:, None], fi2[:, None]), dim=1)
    c3 = torch.cat((filler, x3[:, None], fi3[:, None]), dim=1)

    c = torch.cat((c1[:, None, :], c2[:, None, :], c3[:, None, :]), dim=1)
    c = torch.linalg.det(c) * reciprocaldetJ

    new_us = a + b * new_x + c * new_y
    
    new_us = torch.round(new_us, decimals=3)
    
    new_us = new_us.reshape(new_elementwise_indices.shape[0], new_elementwise_indices.shape[1], -1)
    new_us = torch.where(new_us > 1e-3, new_us, 0)

    ext_elementwise_indices = ext_elementwise_indices.view(ext_new_elementwise_indices.shape[0],
                                                           ext_new_elementwise_indices.shape[1],
                                                           ext_new_elementwise_indices.shape[2],
                                                           -1)[:, :, 0, :]

    return new_us, new_elementwise_indices.long(), ext_elementwise_indices.long()