# data_loader_mis.py (Corrected)

import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import glob
import pickle
import os
from torch_geometric.data import Data as GraphData

class MISConditionalPrefixDataset(Dataset):
    def __init__(self, data_dir, prefix_k_options, prefix_sampling_strategy='scattered'):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.gpickle'))
        self.num_samples = len(self.file_paths)
        self.prefix_k_options = prefix_k_options
        self.prefix_sampling_strategy = prefix_sampling_strategy

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        num_nodes = G.number_of_nodes()
        node_list = sorted(G.nodes())

        # Ground truth labels
        gt_labels = torch.tensor([G.nodes[n]['label'] for n in node_list], dtype=torch.long)
        
        # Edges for torch_geometric
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

        # Determine prefix size
        prefix_fraction = np.random.choice(self.prefix_k_options)
        prefix_k = int(prefix_fraction * num_nodes)
        
        # Get prefix nodes
        perm = torch.randperm(num_nodes)
        prefix_node_indices = perm[:prefix_k]

        # Node state feature
        node_prefix_state = torch.zeros((num_nodes, 1), dtype=torch.float32)
        if prefix_k > 0:
            node_prefix_state[prefix_node_indices] = 1.0

        # Create a torch_geometric Data object
        graph_data = GraphData(
            x=gt_labels.unsqueeze(-1).float(), # Initial feature is just the label
            x_true=gt_labels.clone(), # <<< ADD THIS LINE to store true labels for clamping
            edge_index=edge_index,
            num_nodes=num_nodes,
            prefix_nodes=prefix_node_indices,
            prefix_len=torch.tensor(prefix_k, dtype=torch.long),
            node_prefix_state=node_prefix_state
        )
        return graph_data