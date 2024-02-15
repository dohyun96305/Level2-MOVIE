import argparse

def parse_args() : 

    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--dataset_dir', default = './data/train', type=str)
    parser.add_argument('--outputs_dir', default= './outputs', type=str)
    parser.add_argument('--submits_dir', default = './submits', type = str)
    parser.add_argument('--seed', default = 1337, type = int)
    parser.add_argument("--model", default = 'RecVae', type = str, help = "")


    # preprocessing
    parser.add_argument('--min_users_per_item', type=int, default=0)
    parser.add_argument('--heldout_users', default= 0, type=int)

    # run
    parser.add_argument('--hidden_dim', type=int, default=600)
    parser.add_argument('--latent_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--n_enc_epochs', type=int, default=3)
    parser.add_argument('--n_dec_epochs', type=int, default=1)
    parser.add_argument('--not_alternating', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)') # optimizer 설정
    
    args = parser.parse_args()

    return args