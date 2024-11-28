import os, yaml, time, argparse, _init_paths, copy
from pathlib import Path
from utils.input_output import load_json
from IPython import embed
from pprint import pformat

import torch
torch.set_num_threads(1)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default=None,
                        help='config file path')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--train_type', type=str, default=None, help='')
    parser.add_argument('--tg_domain_split', type=str, default=None, help='')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()

def main(kargs):
    import logging
    import numpy as np
    import random

    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    args = load_json(kargs.cfg)
    from runners import MainRunner
    similarity_list = None
    print('Initializing Model and Dataloader', end='')
    runner = MainRunner(args, similarity_list)
    print('\n'*2)

    print('#' * 100)
    kargs.resume = os.path.join('./ckpt/charades/zero_shot.ckpt')
    setting_msg = 'Zero-Shot Inference'
    print('Evaluation on {} ({})'.format(setting_msg, 'Charades Dataset'))
    print('#' * 100)
    runner._load_model(kargs.resume)
    runner.eval(setting_msg='Zero-Shot Inference')
    print('\n'*2)


    print('#' * 100)
    kargs.resume = os.path.join('./ckpt/charades/unsup.ckpt')
    setting_msg = 'Unsupervised Setting'
    print('Evaluation on {} ({})'.format(setting_msg, 'Charades Dataset'))
    print('#' * 100)
    runner._load_model(kargs.resume)
    runner.eval(setting_msg='Unsupervised Learning')
    print('\n'*2)

    print('#' * 100)
    kargs.resume = os.path.join('./ckpt/charades/full_sup.ckpt')
    setting_msg = 'Fully Supervised Setting'
    print('Evaluation on {} ({})'.format(setting_msg, 'Charades Dataset'))
    print('#' * 100)
    runner._load_model(kargs.resume)
    runner.eval(setting_msg='Fully Supervised Learning')
    print('\n'*2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
