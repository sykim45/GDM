import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator
from logger import Logger
import torch.backends.cudnn as cudnn
from util import parse_args, model_setup, checkpoint_init, init_seed, data_preparation, get_hms
import numpy as np
import scipy.sparse as ssp
import os
import pickle
import warnings
from torch_sparse import SparseTensor
import wandb
import pdb


def load_models(model, predictor, run, args, emb=None):
     model.load_state_dict(torch.load(os.path.join('best_model',
                                                      args.dataset + "_" + args.gnn_type + "_" + str(
                                                          run) + "_best_model.pt")))


def train_collab(model, predictor, data, split_edge, optimizer, batch_size, dataset, args=None):
    model.cuda()
    model.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        
        if hasattr(model, "n_steps"):
            diffusion_loss = model.diffusion(data.x, data.adj_t)
            if args.gnn_type.endswith("OURS") or args.gnn_type.endswith("OURS_ITER"):
                h = model.gnn(data.x, data.adj_t)
            else:
                pass
        else:
            h = model.gnn(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        
        pos_out = model.predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        neg_edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                                 device=h.device)
        neg_out = model.predictor(h[neg_edge[0]], h[neg_edge[1]])

        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = (pos_loss + neg_loss) 

        if hasattr(model, "n_steps"):
            loss += diffusion_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.gnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_collab(model, predictor, data, split_edge, evaluator, batch_size, args):
    model.eval()
    model.gnn.eval()
    model.predictor.eval()  #

    if args.gnn_type.endswith("OURSADJ"):
        h = model.gnn(data.adj_t, data.num_nodes, data.adj_t.storage.value())
    elif args.gnn_type.endswith("OURSFEAT"):
        h = model.gnn(data.x)
    elif args.gnn_type.endswith("OURS"):
        h = model.gnn(data.x, data.adj_t)
    else:
        h = model.gnn(data.x, data.adj_t)
    
    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [model.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [model.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [model.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [model.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [model.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    torch.cuda.empty_cache()
    return results

def train(model, predictor, split_edge, optimizer, batch_size, dataset, data=None, x=None, adj_t=None, args=None):
    return train_collab(model, predictor, data, split_edge, optimizer, batch_size, dataset, args)


def test(model, predictor, split_edge, evaluator, batch_size, dataset, data=None, x=None, adj_t=None, args=None):
    return test_collab(model, predictor, data, split_edge, evaluator, batch_size, args=args)

def trainval_pcc(data, split_edge, model, predictor, optimizer, evaluator, args, run, epoch=None):
    loss = train(model, predictor, split_edge, optimizer,
                 args.batch_size, args.dataset, data=data, args=args)
    if epoch is not None:
        print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
    results = test(model, predictor, split_edge, evaluator,
                   args.batch_size, args.dataset, data=data, args=args)
    return loss, results


def reset_init_optimize(model, predictor, args, emb=None):
    if emb is not None:
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        model.predictor.reset_parameters()
        model.gnn.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.gnn.parameters()) + list(emb.parameters()) +
            list(model.predictor.parameters()), lr=args.lr)
    else:
        model.reset_parameters()
        model.predictor.reset_parameters()
        model.gnn.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.gnn.parameters()) + list(model.predictor.parameters()),
            lr=args.lr)
    return optimizer
