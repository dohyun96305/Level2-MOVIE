import os
import argparse
import pandas as pd
import warnings
import torch

from args import parse_args
from src.utils import Setting, models_load, recall
from src.train.trainer import train
from src.data_preprocess.RecVae_data import RecVae_dataloader, RecVae_train_data, get_data

from inference import result

warnings.filterwarnings('ignore')

def main(args) : 
    Setting.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data! ---------------')
    
    data = RecVae_dataloader(args)
    RecVae_train_data(args, data)

    print(f'--------------- {args.model} Load Data DONE! ---------------')
  

    ######################## DATA PREPROCESS
    print(f'--------------- {args.model} Data PREPROCESSING! ---------------')
    
    data = get_data(args)
    data, = data

    print(f'--------------- {args.model} Data PREPROCESSING DONE!---------------')

    ######################## MODEL LOAD
    print(f'--------------- {args.model} MODEL LOAD---------------')

    model, model_best = models_load(args, data, device)

    print(f'--------------- {args.model} MODEL LOAD DONE!---------------')

    ####################### TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')

    metrics = [{'metric': recall, 'k': 10}]
    model_best = train(args, device, model, model_best, data, metrics)

    print(f'--------------- {args.model} TRAINING DONE! ---------------')

    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')

    result(args, device, model_best, data)

    print(f'--------------- {args.model} PREDICT DONE!!!!!!!!!!---------------')

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name = args.outputs_dir, exist_ok = True)
    os.makedirs(name = args.submits_dir, exist_ok = True)
    main(args = args)