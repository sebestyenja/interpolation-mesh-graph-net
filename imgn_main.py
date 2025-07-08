import torch

from normalizer import cumulative_normalizer as norm
from dataset import static_dataset
from graph_network import graph_network
from trainer import train_cycle

torch.cuda.init()
print("Is CUDA available: " + str(torch.cuda.is_available())) 
print(torch.__version__)
torch.backends.cudnn.benchmark = True

device = "cuda:0"
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(device)

ds_path = "F:/static_dataset/static_crack_conversion/"
data_file = "train.pkl"
test_file = "test.pkl"
# test_file = "valid.pkl"
saving_path = "./"

mode = ["MGN", "INTERPOLATION", "MULTISCALE"][0]
print("Graph model:", mode)

element_order = ["1", "2", "3", "4", "5", "hq"][4]
print("Mesh order:", element_order)

test_example_idx = 12
n_epochs = 100

lr = 1e-4
training_noise = 1e-4

dry_run_flag = False

def main():
    print("Training starts!")

    normalizer_pack = [norm("node_normalizer", 6, 1e6, 1e-6, saving_path, device),
                       norm("edge_normalizer", 3, 1e6, 1e-6, saving_path, device),
                       norm("target_normalizer", 3, 1e6, 1e-6, saving_path, device), 
                       norm("coarse_edge_normalizer", 3, 1e6, 1e-6, saving_path, device)]
    
    test_set = static_dataset(ds_path, data_file, training_noise, mode, element_order, normalizer_pack, device=device)     
    test_set.load_dataset(test_set.ds_path + test_file, test_example_idx)
    test_set.noise_level = 0.0

    data_set = static_dataset(ds_path, data_file, training_noise, mode, element_order, normalizer_pack, device=device)   
    data_set.load_dataset(data_set.ds_path + data_file)
        
    graph = graph_network(linear_depth=4, conv_depth=15, node_input_w=6, edge_input_w=3, hidden_w=128, output_w=3, mode=mode)
        
    train_cycle(graph, data_set, test_set, n_epochs, lr)
            
    print("End of program!")
       
main()
