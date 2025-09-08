import torch
import torch.backends.cudnn as cudnn
from util import parse_args, model_setup, checkpoint_init, init_seed, data_preparation, get_hms, print_endtime
from tools import load_models, train, test
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
from networks.model.link_predictor import LinkPredictor
import os
import wandb
import numpy as np
import time
import pdb


def main(args=None):
    parser, cfg = parse_args(args)
    random_seed = 0
    # 1. Start a new run
    device = torch.device('cuda')
    use_cuda = torch.cuda.is_available()

    # >>> Data
    print("\n[Phase 1]: Data Preparation for {}.".format(parser.dataset))
    if parser.dataset.endswith('ddi'):
        split_edge, adj_t = data_preparation(parser)
    else:
        split_edge, data = data_preparation(parser)

    print('\n[Phase 2] : Model setup')
    net, checkpoint = model_setup(adj_t, parser) if parser.dataset.endswith('ddi') else model_setup(data, parser)
    init_seed(random_seed)
    print(net)
    predictor = LinkPredictor(parser.hidden, parser.hidden, 1,
                              parser.mlp_num_layers, parser.dropout)

    # Embedding for DDI dataset
    if parser.dataset.endswith('ddi'):
        emb = torch.nn.Embedding(adj_t.size(0),
                                 parser.hidden).to(device)

    evaluator = Evaluator(name=str(parser.dataset))
    

    loggers = {
        'Hits@10': Logger(parser.runs, parser),
        'Hits@50': Logger(parser.runs, parser),
        'Hits@100': Logger(parser.runs, parser),
    }

    if use_cuda:
        net = net.to(device)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if not parser.testOnly:
        # >>> Train Model
        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(parser.epochs))
        print('| Initial Learning Rate = ' + str(parser.lr))
        print('| Optimizer = ' + str(cfg['SOLVER']['OPTIMIZER']))

        if parser.dataset.endswith('collab'):
            metric = 'Hits@50'

        total_val, total_test = [], []

        runs = wandb.config.runs if parser.wandb else parser.runs
        best_valid_performance = 0
        best_valid_test_performance = 0
        for run in range(runs):
            #if i > 0:
            init_seed(random_seed=run)
            # os.environ['PYTHONHASHSEED'] = str(i)
            cudnn.deterministic = True
            cudnn.benchmark = False

            net.reset_parameters()
            #predictor.reset_parameters()
            
            optimizer = torch.optim.Adam(
                        list(net.parameters()) , lr=parser.lr)
            
            elapsed_time = 0
            for epoch in range(parser.epochs):
                start_time = time.time()

                if hasattr(net, "drop_p"):
                    net.state_step_reset()
                    
                loss = train(net, predictor, split_edge, optimizer, parser.batch_size, parser.dataset, data=data, args=parser)
                if parser.dataset.endswith('citation2'):
                    print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

                if (epoch+1) % parser.eval_steps == 0:
                    results = test(net, predictor, split_edge, evaluator,
                                    parser.batch_size, parser.dataset, data=data, args=parser)
                    ## Empty CUDA
                    torch.cuda.empty_cache()

                    for key, result in results.items():
                        loggers[key].add_result(run, result)
                    
                    if (epoch + 1) % parser.log_steps == 0:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            print(f'{key}, '
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch + 1:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        #print(torch.softmax(net.alpha.detach().cpu(),dim=0))
                
                epoch_time = time.time() - start_time
                elapsed_time += epoch_time
                print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
                print('---')
                
                valid_performance = results[metric][1]
                if valid_performance > best_valid_performance:
                    if parser.wandb:
                        torch.save(net.state_dict(), os.path.join(wandb.run.dir,parser.dataset + "_" + parser.gnn_type + "_" + str(run) + "_best_model.pt"))
                        #torch.save(predictor.state_dict(), os.path.join(wandb.run.dir, parser.dataset + "_" + parser.gnn_type + "_" + str(run) + "_best_predictor.pt"))
                        if parser.dataset.endswith('ddi'):
                            torch.save(emb.state_dict(), os.path.join(wandb.run.dir,
                                                                            parser.dataset + "_" + parser.gnn_type + "_" + str(
                                                                                run) + "_best_emb.pt"))
                    else:
                        torch.save(net.state_dict(),
                                    os.path.join('best_model', parser.dataset + "_" + parser.gnn_type + "_" + str(run) + '_best_model.pt'))
                        #torch.save(predictor.state_dict(),
                            #          os.path.join('best_model', parser.dataset + "_" + parser.gnn_type + "_" + str(run) + '_best_predictor.pt'))
                        if parser.dataset.endswith('ddi'):
                            torch.save(emb.state_dict(),
                                        os.path.join('best_model',
                                                    parser.dataset + "_" + parser.gnn_type + "_" + str(
                                                        run) + '_best_emb.pt'))
                    best_valid_performance = valid_performance
                    best_valid_test_performance = results[metric][2]
            total_val.append(best_valid_performance)
            total_test.append(best_valid_test_performance)

            for key in loggers.keys():
                print(key)
                final_test, highest_valid = loggers[key].print_statistics(run)
                if parser.wandb:
                    # 4. Log metrics to visualize performance
                    wandb.log({key + '_test': final_test})
                    wandb.log({key + '_valid': highest_valid})
        print_endtime(time.time())        

    print(f'=======All runs:{parser.runs}=======')
    r, rt = 100 * torch.tensor(total_val), 100 * torch.tensor(total_test)
    print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
    print(f'   Final Test: {rt.mean():.2f} ± {rt.std():.2f}')

    for key in loggers.keys():
        print(key)
        final_mean, final_std = loggers[key].print_statistics()
        if parser.wandb:
            wandb.run.summary[key + "_Mean"] = final_mean
            wandb.run.summary[key + "_Std"] = final_std

    elif parser.testOnly:
        if parser.num_run is None:
            results = []
            for run in range(parser.runs):
                load_models(net, predictor, run, parser)
                result= test(net, predictor, split_edge, evaluator, parser.batch_size, parser.dataset, data=data, args=parser)
            results.append(result)
        else:
            else:
            load_models(net, predictor, run, parser)
            results = test(net, predictor, split_edge, evaluator, parser.batch_size, parser.dataset, data=data)
        print(results)


if __name__ == "__main__":
    main()
