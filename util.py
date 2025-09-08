import math
import argparse
import os
import yaml
import torch
from networks.network import getNetwork
import torch.nn.init as init
import numpy as np
import random
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import pdb
from typing import Any
from torch import Tensor
from time import ctime

######################################################################################
# >>> function for arguments
######################################################################################
def parse_args(args):
    parser = argparse.ArgumentParser(description='Diffusion Probabilistic Model for Graph Link Prediction')
    parser.add_argument('--gnn_type', type=str, help='[gcn, gat, gvae, sdiffusion]')
    parser.add_argument('--depth', type=int, default=3, help='depth of model')
    parser.add_argument('--hidden', type=int, default=256, help='hidden_dimension')
    parser.add_argument('--out_feat', type=int, default=128)
    parser.add_argument('--dataset', type=str, help='[ogbl-collab]')
    parser.add_argument('--epochs', type=int, default=200, required=False, help='can omit on test only')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--pred_dropout', type=float, default=0.0)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--pred_hidden', type=int, default=256)
    parser.add_argument('--n_steps', type=int, required=False, default=6))
    parser.add_argument('--iter', '-i', action='store_true', help='iterative diffusion for our model')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_run',type=int,required=False,help='Only needed for testOnly')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--feat_weight', type=float, default=0.03, required=False)
    parser.add_argument('--lap_weight', type=float, default=0.02, required=False)
    parser.add_argument('--recon_weight', type=float, default=1.0, required=False)
    parser.add_argument('--feat_weight2', type=float, default=0.25, required=False)
    parser = parser.parse_args(args)
    cfg = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    return parser, cfg

######################################################################################
# >>> Model setup
######################################################################################
def model_setup(data, args):
    checkpoint = {} # checkpoint dictionary
    print('| Building net type [{}-{}-{}]...'.format(args.dataset, args.gnn_type, args.depth))
    net, file_name = getNetwork(data, args)

    checkpoint['net'] = net.state_dict()
    checkpoint['val_acc'] = 0
    checkpoint['test_acc'] = 0
    checkpoint['ap'] = 0
    checkpoint['epoch'] = 1
    checkpoint['dataset'] = args.dataset
    checkpoint['file_name'] = file_name
    checkpoint['val_total_acc'] = 0
    checkpoint['test_total_acc'] = 0
    return net, checkpoint

def checkpoint_init(checkpoint):
    checkpoint['val_acc'] = 0
    checkpoint['test_acc'] = 0
    checkpoint['ap'] = 0
    checkpoint['epoch'] = 1
    checkpoint['val_total_acc'] = 0
    checkpoint['test_total_acc'] = 0
    return checkpoint


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def print_endtime(time):
    tm = ctime(time)
    print('| End time : {}'.format(tm))
######################################################################################
# >>> Dataset setup
######################################################################################

def data_preparation(args):
    dataset = args.dataset
    #device = args.device
    device = torch.device('cuda')
    if dataset.endswith('ddi'):
        dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        adj_t = data.adj_t.to(device)

        split_edge = dataset.get_edge_split()

        # We randomly pick some training samples that we want to evaluate on:
        torch.manual_seed(12345)
        idx = torch.randperm(split_edge['train']['edge'].size(0))
        idx = idx[:split_edge['valid']['edge'].size(0)]
        split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

        return split_edge, adj_t
    else:
        if dataset.endswith('citation2'):
            dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                             transform=T.ToSparseTensor())
            data = dataset[0]
            data.adj_t = data.adj_t.to_symmetric()
            data = data.to(device)

            split_edge = dataset.get_edge_split()

            # We randomly pick some training samples that we want to evaluate on:
            torch.manual_seed(12345)
            idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
            split_edge['eval_train'] = {
                'source_node': split_edge['train']['source_node'][idx],
                'target_node': split_edge['train']['target_node'][idx],
                'target_node_neg': split_edge['valid']['target_node_neg'],
            }

        elif dataset.endswith('collab'):
            dataset = PygLinkPropPredDataset(name='ogbl-collab')
            data = dataset[0]
            edge_index = data.edge_index
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor(remove_edge_index=False)(data)

            split_edge = dataset.get_edge_split()

            # Use training + validation edges for inference on test set.
            if args.use_valedges_as_input:
                val_edge_index = split_edge['valid']['edge'].t()
                full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
                data.full_adj_t = data.full_adj_t.to_symmetric()
            else:
                data.full_adj_t = data.adj_t

            data = data.to(device)

        elif dataset.endswith('ppa'):
            dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                             transform=T.ToSparseTensor(remove_edge_index=False))

            data = dataset[0]
            data.x = data.x.to(torch.float)
            if args.use_node_embedding:
                data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
            data = data.to(device)

            split_edge = dataset.get_edge_split()

            #Pre-compute GCN Normalization
            if args.gnn_type == 'GCN':
                adj_t = data.adj_t.set_diag()
                deg = adj_t.sum(dim=1).to(torch.float)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
                data.adj_t = adj_t

        return split_edge, data

def data_preparation_node(args, config):
    data_name = str(args.dataset)
    device = torch.device('cuda')
    dataset = PygNodePropPredDataset(name=data_name, transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    setattr(data, 'num_classes', config['DATASET'][data_name]['num_classes'])
    if data_name.endswith('arxiv'):
        data.adj_t = data.adj_t.to_symmetric()
    
    elif data_name.endswith('products'):
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
        
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    
    return data.to(device), split_idx, train_idx


def init_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(random_seed)

