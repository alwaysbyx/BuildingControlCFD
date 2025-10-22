#!/usr/bin/env python
#-*- coding:utf-8 _*-
import sys
import os
sys.path.append('../..')
sys.path.append('..')
import math
from typing import List

import re
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")

from args import get_args
from data_utils import get_model, get_loss_func, collate_op, MIODataLoader, MIODataset, LogLoss
from utils import get_seed, get_num_params
from models.optimizer import Adam, AdamW



'''
    A general code framework for training neural operator on irregular domains
'''

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']






def train(model, loss_func, metric_func,
              train_loader, valid_loader,
              optimizer, lr_scheduler,
              epochs=10,
              writer=None,
              device="cuda",
              patience=10,
              grad_clip=0.999,
              start_epoch: int = 0,
              print_freq: int = 20,
              model_save_path='./data/checkpoints/',
              save_mode='state_dict',  # 'state_dict' or 'entire'
              model_name='model.pt',
              result_name='result.pt',
              args=None):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    result = None
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)for s in EPOCH_SCHEDULERS)
    t = time.time()
    for epoch in range(start_epoch, end_epoch):
        model.train()
        torch.cuda.empty_cache()
        for batch in train_loader:

            loss = train_batch(model, loss_func, batch, optimizer, lr_scheduler, device, grad_clip=grad_clip)

            loss = np.array(loss)
            loss_epoch.append(loss)
            it += 1
            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)
            log = f"epoch: [{epoch+1}/{end_epoch}]"
            if loss.ndim == 0:  # 1 target loss
                _loss_mean = np.mean(loss_epoch)
                log += " loss: {:.6f}".format(_loss_mean)
            else:
                _loss_mean = np.mean(loss_epoch, axis=0)
                for j in range(len(_loss_mean)):
                    log += " | loss {}: {:.6f}".format(j, _loss_mean[j])
            log += " | current lr: {:.3e}".format(lr)

            if it % print_freq==0:
                print(log)

            if writer is not None:
                for j in range(len(_loss_mean)):
                    writer.add_scalar("train_loss_{}".format(j),_loss_mean[j], it)    #### loss 0 seems to be the sum of all loss


        print(_loss_mean)
        loss_train.append(_loss_mean)
        loss_epoch = []

        val_result = validate_epoch(model, metric_func, valid_loader, device)

        loss_val.append(val_result["metric"])
        val_metric = val_result["metric"].mean()
        print(val_metric)
        s = time.time()
        wandb.log(data=dict(
            train_err = _loss_mean[0], 
            train_metric = _loss_mean[-1], 
            test_err = val_metric,
            time =  s - t
        ),
            step=epoch+1,
            commit=True)
            

        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric
            best_model_state_dict = model.state_dict()


        if lr_scheduler and is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(val_metric)
            else:
                lr_scheduler.step()


        if val_result["metric"].size == 1:
            log = "| val metric 0: {:.6f} ".format(val_metric)

        else:
            log = ''
            for i, metric_i in enumerate(val_result['metric']):
                log += '| val metric {} : {:.6f} '.format(i, metric_i)

        if writer is not None:
            if val_result["metric"].size == 1:
                writer.add_scalar('val loss {}'.format(metric_func.component),val_metric, epoch)
            else:
                for i, metric_i in enumerate(val_result['metric']):
                    writer.add_scalar('val loss {}'.format(i), metric_i, epoch)


        log += "| best val: {:.6f} at epoch {} | current lr: {:.3e}".format(best_val_metric, best_val_epoch+1, lr)

        desc_ep = ""
        if _loss_mean.ndim == 0:  # 1 target loss
            desc_ep += "| loss: {:.6f}".format(_loss_mean)
        else:
            for j in range(len(_loss_mean)):
                if _loss_mean[j] > 0:
                    desc_ep += "| loss {}: {:.3e}".format(j, _loss_mean[j])

        desc_ep += log
        print(desc_ep)

        result = dict(
            best_val_epoch=best_val_epoch,
            best_val_metric=best_val_metric,
            loss_train=np.asarray(loss_train),
            loss_val=np.asarray(loss_val),
            lr_history=np.asarray(lr_history),
            best_model=best_model_state_dict,
            optimizer_state=optimizer.state_dict()
        )
        pickle.dump(result, open(os.path.join(model_save_path, result_name),'wb'))
        if epoch % 10 == 0:
            result['args']= args
            checkpoint = {'args':args, 'model':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(checkpoint, os.path.join('./data/checkpoints/{}'.format(model_path)))
    return result




def train_batch(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.999):
    optimizer.zero_grad()

    g, u_p, g_u = data

    g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)


    out = model(g, u_p, g_u)


    y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
    loss, reg,  metric = loss_func(g, y_pred, y)
    loss = loss + reg
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


    if lr_scheduler:
        lr_scheduler.step()



    return (loss.item(), reg.item(), np.mean(metric))



def validate_epoch(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            g, u_p, g_u = data
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

            out = model(g, u_p, g_u)

            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            _, _, metric = metric_func(g, y_pred, y)

            metric_val.append(metric)
    return dict(metric=np.mean(metric_val, axis=0))


class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions
    """
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        """

        Parameters
        ----------
        d : int, optional
            dimension of data on which to compute, by default 1
        p : int, optional
            order of L-norm, by default 2
            L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
        L : float or list, optional
            quadrature weights per dim, by default 2*math.pi
            either single scalar for each dim, or one per dim
        reduce_dims : int, optional
            dimensions across which to reduce for loss, by default 0
        reductions : str, optional
            whether to reduce each dimension above 
            by summing ('sum') or averaging ('mean')
        """
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            allowed_reductions = ["sum", "mean"]
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean',\
                f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions}"
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean',\
                        f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions[j]}"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_h(self, x):
        """uniform_h creates default normalization constants
        if none already exist.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        h : list
            list of normalization constants per-dim
        """
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        """
        reduce x across all dimensions in self.reduce_dims 
        according to self.reductions

        Params
        ------
        x: torch.Tensor
            inputs
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constants for reduction
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/||y||

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


if __name__ == "__main__":
    args = get_args()
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
        device = torch.device("cpu")
    print(device)
    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)
    
    # Insert your wandb setting
    # wandb_init_args = {}
    # wandb_name = f"GNOT{args.seed}"
    # wandb.login(key="Your Key")
    # wandb_init_args = dict(name=wandb_name)
    # wandb.init(**wandb_init_args)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path, test_path = os.path.join(BASE_DIR, "dataset/train_data_norm.pkl"), os.path.join(BASE_DIR, "dataset/test_data_norm.pkl")
    dataset_name = "co2"
    train_dataset = MIODataset(train_path, name=dataset_name, train=True, train_num=args.train_num,
                               sort_data=args.sort_data,
                               normalize_y=args.use_normalizer,
                               normalize_x=args.normalize_x)
    test_dataset = MIODataset(test_path, name=dataset_name, train=False, test_num=args.test_num,
                              sort_data=args.sort_data,
                              normalize_y=args.use_normalizer,
                              normalize_x=args.normalize_x, y_normalizer=train_dataset.y_normalizer,
                              x_normalizer=train_dataset.x_normalizer, up_normalizer=train_dataset.up_normalizer)
    args.dataset_config = train_dataset.config
    args.dataset = "co2"

    train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = MIODataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    args.space_dim = int(re.search(r'\d', args.dataset).group())
    args.normalizer =  train_dataset.y_normalizer.to(device) if train_dataset.y_normalizer is not None else None

    #### set random seeds
    get_seed(args.seed)
    torch.cuda.empty_cache()

    loss_func = LogLoss(normalizer=args.normalizer)
    metric_func = LogLoss(normalizer=args.normalizer)

    metric_func = get_loss_func(name='rel2', args=args, regularizer=False, normalizer=args.normalizer)

    model = get_model(args)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")


    path_prefix = args.dataset  + '_{}_'.format(args.component) + model.__name__ + "local_quantile" + time.strftime('_%m%d_%H_%M_%S')
    model_path, result_path = path_prefix + '.pt', path_prefix + '.pkl'

    print(f"Saving model and result in ./../models/checkpoints/{model_path}\n")


    if args.use_tb:
        writer_path =  './data/logs/' + path_prefix
        log_path = writer_path + '/params.txt'
        writer = SummaryWriter(log_dir=writer_path)
        fp = open(log_path, "w+")
        sys.stdout = fp

    else:
        writer = None
        log_path = None


    print(model)
    # print(config)

    epochs = args.epochs
    lr = args.lr


    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9,0.999))
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9, 0.999))
    else:
        raise NotImplementedError



    if args.lr_method == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=epochs)
    elif args.lr_method == 'step':
        print('Using step learning rate schedule')
        scheduler = StepLR(optimizer, step_size=args.lr_step_size*len(train_loader), gamma=0.7)
    elif args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = LambdaLR(optimizer, lambda steps: min((steps+1)/(args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader)/float(steps + 1), 0.5)))


    time_start = time.time()

    result = train(model, loss_func, metric_func,
                       train_loader, test_loader,
                       optimizer, scheduler,
                       epochs=epochs,
                       grad_clip=args.grad_clip,
                       patience=None,
                       model_name=model_path,
                       model_save_path='./data/checkpoints/',
                       result_name=result_path,
                       writer=writer,
                       device=device,
                       args=args)

    print('Training takes {} seconds.'.format(time.time() - time_start))

    model.eval()
    val_metric = validate_epoch(model, metric_func, test_loader, device)
    print(f"\nBest model's validation metric in this run: {val_metric}")




