import pandas as pd
import numpy as np
import os

import bottleneck as bn

from src.train.trainer import generate

def result(args, device, model, data_in, samples_perc_per_epoch=1):
    model.eval()
    items=[]
    user = pd.read_csv(os.path.join(args.outputs_dir, 'unique_uid.csv'), header=None)
    item = pd.read_csv(os.path.join(args.outputs_dir, 'unique_sid.csv'), header=None)
    item = item.to_numpy()

    for batch in generate(batch_size = args.batch_size,
                        device = device,
                        data_in = data_in,
                        samples_perc_per_epoch = samples_perc_per_epoch
                        ):

        ratings_in = batch.get_ratings_to_dev()

        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()

        ratings_pred[batch.get_ratings().nonzero()] = -np.inf


        batch_users = ratings_pred.shape[0]
        idx = bn.argpartition(-ratings_pred, 10, axis=1)
        X_pred_binary = np.zeros_like(ratings_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :10]] = True

        for i in X_pred_binary:
            items.append(item[i])

    users = np.array(user)
    users = users.repeat(10).reshape(-1,1)
    items = np.array(items).reshape(-1,1)

    result = np.concatenate((users,items),axis=1)
    result = pd.DataFrame(result, columns=['user','item'])
    result.to_csv(os.path.join(args.submits_dir, 'result.csv'), index=False)
