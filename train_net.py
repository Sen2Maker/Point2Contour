import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import network, data
from training import train_n
from tools import load_and_merge_config

os.environ["PYTHONHASHSEED"] = "2024"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  
import torch
import random
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.benchmark = False  
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
np.random.seed(2024)
random.seed(2024)

def make_args_parser():
    parser = argparse.ArgumentParser("Point2Contour training")

    parser.add_argument('-c', '--config', default="config/default_model.yaml", type=str, help='Configuration file')

    parser.add_argument('--num_epochs', default='90', type=int, help='Number of epochs')
    parser.add_argument('--ckpt_epoch', default='10', type=int, help='Checkpoint interval')
    parser.add_argument('--steps_summary', default='20', type=int, help='Log interval')

    parser.add_argument('--resume',default=False, action="store_true", help='Resume training')
    parser.add_argument('--resume_dir', default=None, type=str, help='Checkpoint directory')

    parser.add_argument('--data_root', required=True, type=str, help='Processed dataset directory')

    parser.add_argument('--num_batch', default='1', type=int, help='Batch size')
    parser.add_argument('--num_workers', default='8', type=int, help='Data loader workers')
    parser.add_argument('--pin_memory', default=True, help='Enable pinned memory')
    parser.add_argument('--shuffle', default=True, help='Shuffle training data')
    parser.add_argument('--prefetch_factor', default='8', type=int, help='Data loader prefetch factor')

    args = parser.parse_args()
    return args


def run_one_training(opt):
    print(f"Starting training: resume={opt.resume}.")
    print(f"Output directory: {opt.log_path}")

    if not opt.resume:
        import shutil
        os.makedirs(opt.log_path, exist_ok=True)

        shutil.copyfile(opt.source_config_path, os.path.join(opt.log_path, 'config.yaml'))

    train_dataloader = data.get_dataloader(opt, dataset_mode='train')
    opt['train_dataloader'] = train_dataloader

    val_dataloader = data.get_dataloader(opt, dataset_mode='val')
    opt['val_dataloader'] = val_dataloader

    model = network.define_model(opt)
    model.cuda()

    train_n(opt, model)


if __name__ == '__main__':

    args = make_args_parser()

    opt = load_and_merge_config(args)

    run_one_training(opt)
