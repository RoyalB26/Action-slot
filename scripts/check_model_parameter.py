from model import generate_model
import sys

from get_parser import get_parser
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args, logdir = get_parser()
    print(args)
    logdir = logdir.replace(':', '_').replace('\n', '_')
    seq_len = args.seq_len

    num_ego_class = 4
    num_actor_class = 64
    if args.taco_class == 'Action':
        num_actor_class = 20
    elif args.taco_class == 'Object':
        num_actor_class = 6

    model= generate_model(args, num_ego_class, num_actor_class)
    