# based on https://github.com/dawenl/vae_cf

import numpy as np
import random
import torch
import os

import bottleneck as bn

from .models import *

class Setting:
    @staticmethod

    def seed_everything(seed):
     
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def make_dir(self, path):
    
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path

def models_load(args, df, device) : 

    model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': df.shape[1]
    }

    model = VAE(**model_kwargs).to(device)
    model_best = VAE(**model_kwargs).to(device)

    return model, model_best


def recall(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall