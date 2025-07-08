import torch 
import torch.nn as nn

from utils import upscale_by_linear_weights_2D

class graph_network(nn.Module):
    def __init__(self, linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w, mode="MGN"):
        super(graph_network, self).__init__()
        self.name = "graph_network"
        if mode == "MGN":
            self.forward = self.forward_mgn
            self.network = self._build_mgn(linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w)
        elif mode == "INTERPOLATION":
            self.forward = self.forward_interpolation
            self.network = self._build_i_mgn(linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w)
            delattr(self, "edge_encoder")
            delattr(self, "edge_messenger")
        elif mode == "MULTISCALE":
            self.forward = self.forward_multiscale
            self.network = self._build_i_mgn(linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w, multiscale=True)
        else:
            input("PROCESSING MODE ERROR!")
        
    def _build_MLP(self, depth, input_w, output_w, bias=True, input_net=True):
        
        network = nn.ModuleList()
        
        if input_net:
            network.append(nn.Linear(input_w, output_w, bias=bias))
            network.append(nn.LeakyReLU(negative_slope=0.05))
            for i in range(depth - 1):
                network.append(nn.Linear(output_w, output_w, bias=bias))
                network.append(nn.LeakyReLU(negative_slope=0.05))
            network.append(nn.LayerNorm(output_w))
        else:
            for i in range(depth - 1):
                network.append(nn.Linear(input_w, input_w, bias=bias))
                network.append(nn.LeakyReLU(negative_slope=0.05))
            network.append(nn.Linear(input_w, output_w, bias=True))
            
        network = nn.Sequential(*network)
        
        return network
    
    def _build_mgn(self, linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w):   
        
        self.node_encoder = self._build_MLP(linear_depth, node_input_w, hidden_w)    
        self.edge_encoder = self._build_MLP(linear_depth, edge_input_w, hidden_w)
            
        self.node_messenger = nn.ModuleList()
        self.edge_messenger = nn.ModuleList()
        
        for i in range(conv_depth):
            self.node_messenger.append(self._build_MLP(linear_depth, 2 * hidden_w, hidden_w))
            self.edge_messenger.append(self._build_MLP(linear_depth, 3 * hidden_w, hidden_w))
            
        self.decoder = self._build_MLP(linear_depth, hidden_w, output_w, input_net=False)
        
    def _build_i_mgn(self, linear_depth, conv_depth, node_input_w, edge_input_w, hidden_w, output_w, multiscale=False):
        
        node_w_mult = 3 if multiscale else 2
        
        self.node_encoder = self._build_MLP(linear_depth, node_input_w, hidden_w)    
        self.edge_encoder = self._build_MLP(linear_depth, edge_input_w, hidden_w) 
        self.coarse_edge_encoder = self._build_MLP(linear_depth, edge_input_w, hidden_w)
            
        self.node_messenger = nn.ModuleList()
        self.edge_messenger = nn.ModuleList()
        self.coarse_node_messenger = nn.ModuleList()
        self.coarse_edge_messenger = nn.ModuleList()
        
        for i in range(conv_depth):
            self.node_messenger.append(self._build_MLP(linear_depth, node_w_mult * hidden_w, hidden_w))
            self.edge_messenger.append(self._build_MLP(linear_depth, 3 * hidden_w, hidden_w))
            self.coarse_node_messenger.append(self._build_MLP(linear_depth, 3 * hidden_w, hidden_w))
            self.coarse_edge_messenger.append(self._build_MLP(linear_depth, 3 * hidden_w, hidden_w))
            
        self.decoder = self._build_MLP(linear_depth, hidden_w, output_w, input_net=False)
     
    def forward_mgn(self, nodes, edges, edge_idx, coarse_edges, coarse_edge_idx, aggr_weights, aggr_edge_idx, int_weights, int_receivers, int_senders):
        
        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(edges)
        
        for i in range(len(self.node_messenger)):
            
            edges = edges + self.edge_messenger[i](torch.cat((edges, nodes[edge_idx[:, 0]], nodes[edge_idx[:, 1]]), dim=1))
            
            aggr_edges = torch.scatter_add(torch.zeros(nodes.shape, device=nodes.device).detach(),
                                           0,
                                           edge_idx[:, 0, None].expand(-1, nodes.shape[1]).to(nodes.device),
                                           edges.to(nodes.device))
            
            nodes = nodes + self.node_messenger[i](torch.cat((nodes, aggr_edges), dim=1))
        
        return self.decoder(nodes)    
                
    def forward_multiscale(self, nodes, edges, edge_idx, coarse_edges, coarse_edge_idx, aggr_weights, aggr_edge_idx, int_weights, int_receivers, int_senders):
        
        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(edges)
        
        coarse_nodes = nodes[coarse_edge_idx[:, 0].unique()].clone()
        coarse_edges = self.coarse_edge_encoder(coarse_edges)

        for i in range(len(self.node_messenger)):
            
            coarse_edges = coarse_edges + self.coarse_edge_messenger[i](torch.cat((coarse_edges,
                                                                                   coarse_nodes[coarse_edge_idx[:, 0]],
                                                                                   coarse_nodes[coarse_edge_idx[:, 1]]), dim=1))
            
            aggr_nodes = torch.scatter_add(torch.zeros((aggr_edge_idx[:, 0].max() + 1, nodes.shape[1]), device=nodes.device),
                                           0,
                                           aggr_edge_idx[:, 0, None].expand(-1, nodes.shape[1]),
                                           nodes[aggr_edge_idx[:, 1]] * aggr_weights[:, None])

            aggr_coarse_edges = torch.scatter_add(torch.zeros(coarse_nodes.shape, device=coarse_nodes.device).detach(),
                                                  0,
                                                  coarse_edge_idx[:, 0, None].expand(-1, coarse_nodes.shape[1]).to(coarse_nodes.device),
                                                  coarse_edges.to(coarse_nodes.device))
            
            coarse_nodes = coarse_nodes + self.coarse_node_messenger[i](torch.cat((coarse_nodes, aggr_coarse_edges, aggr_nodes), dim=1))
            
            edges = edges + self.edge_messenger[i](torch.cat((edges, nodes[edge_idx[:, 0]], nodes[edge_idx[:, 1]]), dim=1))
                      
            aggr_edges = torch.scatter_add(torch.zeros(nodes.shape, device=nodes.device).detach(),
                                           0,
                                           edge_idx[:, 0, None].expand(-1, nodes.shape[1]).to(nodes.device),
                                           edges.to(nodes.device))
                
            int_coarse_nodes = upscale_by_linear_weights_2D(coarse_nodes,
                                                            int_weights,
                                                            int_receivers,
                                                            int_senders,
                                                            output_node_count=nodes.shape[0])
            
            nodes = nodes + self.node_messenger[i](torch.cat((nodes, aggr_edges, int_coarse_nodes), dim=1))
            
        return self.decoder(nodes) 
    
    def forward_interpolation(self, nodes, edges, edge_idx, coarse_edges, coarse_edge_idx, aggr_weights, aggr_edge_idx, int_weights, int_receivers, int_senders):
        
        nodes = self.node_encoder(nodes)
        
        coarse_nodes = nodes[coarse_edge_idx[:, 0].unique()].clone()
        coarse_edges = self.coarse_edge_encoder(coarse_edges)

        for i in range(len(self.node_messenger)):
            
            coarse_edges = coarse_edges + self.coarse_edge_messenger[i](torch.cat((coarse_edges,
                                                                                   coarse_nodes[coarse_edge_idx[:, 0]],
                                                                                   coarse_nodes[coarse_edge_idx[:, 1]]), dim=1))
            
            aggr_nodes = torch.scatter_add(torch.zeros((aggr_edge_idx[:, 0].max() + 1, nodes.shape[1]), device=nodes.device),
                                           0,
                                           aggr_edge_idx[:, 0, None].expand(-1, nodes.shape[1]),
                                           nodes[aggr_edge_idx[:, 1]] * aggr_weights[:, None])

            aggr_coarse_edges = torch.scatter_add(torch.zeros(coarse_nodes.shape, device=coarse_nodes.device).detach(),
                                                  0,
                                                  coarse_edge_idx[:, 0, None].expand(-1, coarse_nodes.shape[1]).to(coarse_nodes.device),
                                                  coarse_edges.to(coarse_nodes.device))
            
            coarse_nodes = coarse_nodes + self.coarse_node_messenger[i](torch.cat((coarse_nodes, aggr_coarse_edges, aggr_nodes), dim=1))
                
            int_coarse_nodes = upscale_by_linear_weights_2D(coarse_nodes,
                                                            int_weights,
                                                            int_receivers,
                                                            int_senders,
                                                            output_node_count=nodes.shape[0])
            
            nodes = nodes + self.node_messenger[i](torch.cat((nodes, int_coarse_nodes), dim=1))
            
        return self.decoder(nodes) 