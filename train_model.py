"""
    Deep Unsupervised Cardinality Estimation Source Code

    Source Code used as is or modified from the above mentioned source
"""


import argparse
import os
import time

import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import datasets
import made
import pandas as pd

import pickle

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

# Training.
parser.add_argument('--dataset', type=str, default='customer_special', help='Dataset.')
parser.add_argument('--num-gpus', type=int, default=0, help='#gpus.')
parser.add_argument('--bs', type=int, default=2048, help='Batch size.')
parser.add_argument(
    '--warmups',
    type=int,
    default=0,
    help='Learning rate warmup steps.')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help='Number of epochs to train for.')
parser.add_argument('--constant-lr',
                    type=float,
                    default=None,
                    help='Constant LR?')

parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Column masking training, which permits wildcard skipping'\
    ' at querying time.')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=512,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=3, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')


parser.add_argument(
    '--input-encoding',
    type=str,
    default='embed',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='embed',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')
parser.add_argument(
    '--table-special',
    type=str,
    default='special',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')

parser.add_argument('--do-compression', action='store_true', help='Do compression on the columns of the table?')


args = parser.parse_args()
# TODO: needs to be set if we want compression
args.do_compression = True
# pre-configure some parameters to not need to enter them
args.column_masking = True
args.residual = True
args.direct_io = True


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)

        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)

    return ret


def RunEpoch(split,
             model,
             opt,
             train_data,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    nsamples = 1

    for step, xb in enumerate(loader):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]


        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()

            else:
                loss = model.nll(xbhat, xb).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Train Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss-original: {} loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item(),
                            loss.item() / np.log(2), table_bits, lr))
            else:
                print('Test Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

        torch.cuda.empty_cache()
    if return_losses:
        return losses
    return np.mean(losses)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None, col_maximums=None, do_compression=True, table_name=None):

    if col_maximums is None:
        col_maximums = [int(c.data.max()) + 1 for c in cols_to_train]

    model = made.MADE(
        nin=len(col_maximums),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum(col_maximums),
        input_bins=col_maximums,
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    print(f'MODEL IS ON {DEVICE}')

    return model

def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)

def LoadExistingModel(table_name, column_names, folder_name):
    """
    Load an existing AR model.
    :param table_name:
    :param column_names:
    :param folder_name:
    :return:
    """
    all_ckpts = glob.glob('./{}/{}-*.pt'.format(folder_name, table_name))

    selected_model = all_ckpts[0]
    print('selecting model: ')
    print(selected_model)

    z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+)-max(.*)q.*.pt',selected_model)

    seed = int(z.group(3))
    print('original column names for ar model are: ')
    print(column_names)
    # take the maximum values per column which we store as part of the model name
    maximums_per_column = [int(val) for val in z.group(4).split('_')]
    print('max per col')
    print(maximums_per_column)

    layers_size = 512

    order = None

    model = MakeMade(
        scale=layers_size,
        cols_to_train=column_names.split(','),
        seed=seed,
        fixed_ordering=order,
        col_maximums=maximums_per_column,
        table_name=table_name
    )

    model.load_state_dict(torch.load(selected_model, map_location=torch.device(DEVICE)), strict=False)
    model.eval()

    print(model)
    return model


def TrainTask(seed=0, t_spec_name='', data=None, column_names='', dataset_name='table_special', save_folder_name=None,
              save_compressor_folder_name=None):
    torch.manual_seed(0)
    np.random.seed(0)
    # print(t_spec_name)
    args.table_special = t_spec_name
    args.dataset = dataset_name
    print(f"Table name is {args.table_special}")
    print(f"Dataset name is {args.dataset}")
    print(f'Are we doing compression? {args.do_compression}')

    assert args.dataset in ['table_special']
    if args.dataset == 'table_special':
        if args.table_special == '':
            raise ValueError("The argument args.table_special has to be set when using 'table_special'")
        table = datasets.LoadTableSpecial(data, table_name=args.table_special, do_compression=True,
                                          column_names=column_names, folder_name=args.table_special)


    null_value_date = pd.to_datetime('1900-01-01', infer_datetime_format=True).to_datetime64()

    for c in table.columns:
        if 'date' not in c.name:
            table.data[c.name].fillna(value=0)#.groupby(c.name).size()
        else:
            table.data[c.name].fillna(value=null_value_date)#.groupby(c.name).size()


    table_bits = Entropy(
        table,
        table.data.groupby([c.name for c in table.columns]).size(),
        [2])[0]
    fixed_ordering = None

    table_train = table

    if args.dataset in ['table_special']:
        # print(table.compressor_element.)
        # exit(1)
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
    else:
        assert False, args.dataset

    #report information about the model such as the number of neurons in every layer, the size of the model parameters etc..
    mb = ReportModel(model)

    print('Applying InitWeight()')
    model.apply(InitWeight)
    opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    bs = args.bs
    log_every = 5

    train_data = common.TableDataset(table_train)

    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):

        mean_epoch_train_loss = RunEpoch('train',
                                         model,
                                         opt,
                                         train_data=train_data,
                                         val_data=train_data,
                                         batch_size=bs,
                                         epoch_num=epoch,
                                         log_every=log_every,
                                         table_bits=table_bits)

        if epoch % 1 == 0:
            print('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2)))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))

        train_losses.append(mean_epoch_train_loss)

    print('Training done; evaluating likelihood on full data:')
    all_losses = RunEpoch('test',
                          model,
                          train_data=train_data,
                          val_data=train_data[:5],
                          opt=None,
                          batch_size=1024,
                          log_every=500,
                          table_bits=table_bits,
                          return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits
    print('ADDING COMPRESSOR')
    model.compressor_element = table.compressor_element

    with open('./{}/{}_compressor.pickle'.format(save_compressor_folder_name, args.table_special), 'wb') as store_handle:
        pickle.dump(table.compressor_element, store_handle, protocol=pickle.HIGHEST_PROTOCOL)

    if seed is not None:
        PATH = '{}/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-max{}q.pt'.format(
        # PATH = 'models/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-max{}q.pt'.format(
            save_folder_name,
            args.table_special, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed, '_'.join([str(int(c.data.max()) + 1) for c in table.columns]))
    else:
        PATH = '{}/{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-max{}q.pt'.format(
            save_folder_name,
            args.table_special, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed, time.time())

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print('Saved to:')
    print(PATH)
    return model