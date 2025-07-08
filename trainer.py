import torch
import torch.nn as nn
import numpy as np

def train(i, graph, optimizer, loss_fn, data_set):
    
    graph.zero_grad()

    data_pack = data_set.get_data(i)
    
    network_output = graph.forward(data_pack.node_pack.node_features,
                                   data_pack.edge_pack.edge_features, data_pack.edge_pack.edge_idx,
                                   data_pack.edge_pack.coarse_edge_features, data_pack.edge_pack.coarse_edge_idx,
                                   data_pack.edge_pack.aggr_edge_weights, data_pack.edge_pack.aggr_edge_idx,
                                   data_pack.edge_pack.int_edge_weights, data_pack.edge_pack.int_edge_receivers, data_pack.edge_pack.int_edge_senders)
        
    loss = loss_fn(network_output, data_pack.target_pack.normalized_target)
    
    loss.backward()
            
    nn.utils.clip_grad_norm_(graph.parameters(), max_norm=1.0, norm_type=2)
             
    optimizer.step()
        
    return loss.item()

def train_cycle(graph, data_set, test_set, n_epochs, lr):
    global norm_info_accumulation_steps
    
    loss_fn = nn.MSELoss()

    dataset_len = str(data_set.__len__())
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    
    print("Start norm info accumulation!")
    data_set.shuffle()
    for i in range(min(1000, data_set.__len__())):
        _ = data_set.get_data(i)
    
    print("Start the training!")
    loss_log = torch.empty((0,)).detach().cpu()
    for epoch in range(n_epochs):

        print("Start epoch!", epoch)
        data_set.shuffle()
           
        epoch_loss = torch.zeros((data_set.__len__(),), requires_grad=False).detach()

        for i in range(data_set.__len__()):
                        
            loss_average = train(i, graph,  optimizer, loss_fn, data_set) 
            
            epoch_loss[i] = loss_average

            if i % 100 == 0:
                print("Epoch:" , epoch, "/", n_epochs, "- step", str(i).ljust(len(dataset_len)), "/", dataset_len, "- Loss:", "{:.8f}".format(loss_average))
                
        lr_scheduler.step()
            
        loss_log = torch.cat((loss_log, epoch_loss.detach().cpu()), dim=0)

        np_loss_log = loss_log.detach().cpu().numpy()

        np.save("./losses.npy", np_loss_log)