import numpy as np

import torch
from torch import optim

import random
from copy import deepcopy

import pandas as pd
import bottleneck as bn
from importlib import import_module

class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1

    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

def run(args, device, model, opts, train_data, n_epochs, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=args.batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            _, loss = model(ratings, beta=args.beta, gamma=args.gamma, dropout_rate=dropout_rate)
            loss.backward()

            for optimizer in opts:
                optimizer.step()

def evaluate(args, device, model, data_in, data_out, metrics, samples_perc_per_epoch=1):
    metrics = deepcopy(metrics)
    model.eval()

    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size = args.batch_size,
                          device = device,
                          data_in = data_in,
                          data_out = data_out,
                          samples_perc_per_epoch = samples_perc_per_epoch
                         ):

        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)

        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()

        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf

        for m in metrics:
            m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()

    return [x['score'] for x in metrics]

def train(args, device, model, model_best, df, metrics) : 

    best_recall = -np.inf
    train_scores, valid_scores = [], []

    learning_kwargs = {
        'args' : args, 
        'device' : device, 
        'model': model,
        'train_data': df,
    }

    decoder_params = set(model.decoder.parameters())
    encoder_params = set(model.encoder.parameters())

    opt_encoder_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    opt_decoder_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer_encoder = opt_encoder_module(
            encoder_params,
            lr=args.lr,
        )

    optimizer_decoder = opt_decoder_module(
            decoder_params,
            lr=args.lr,
        )


    for epoch in range(args.n_epochs):

        if args.not_alternating:
            run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
        else:
            run(opts=[optimizer_encoder], n_epochs=args.n_enc_epochs, dropout_rate=0.5, **learning_kwargs)
            model.update_prior()
            run(opts=[optimizer_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)

        train_scores.append(
            evaluate(args, device, model, df, df, metrics, 0.01)[0]
        )

        if train_scores[-1] > best_recall:
            best_recall = train_scores[-1]
            model_best.load_state_dict(deepcopy(model.state_dict()))


        print(f'epoch {epoch} | train recall@10: {train_scores[-1]:.4f}')


    return model_best
