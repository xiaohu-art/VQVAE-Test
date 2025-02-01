import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # If the process hangs, set this to false: https://github.com/huggingface/transformers/issues/5486.
os.environ['TMPDIR'] = '/home/ubuntu/Desktop/VQ-VAE-Test/tmp'
tempfile.tempdir = '/home/ubuntu/Desktop/VQ-VAE-Test/tmp'

import sys
import torch


import torch.distributed as dist
import torch.multiprocessing as mp

from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP


from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.data.dataloader import get_dataloader
from src.eval_utils.eval_loop import eval_loop
from src.train_utils.train_loop import train_loop
from src.train_utils.trainer import Train_Manager
from src.train_utils.utils import train_parser, get_model, log_codebook_usage, get_opt
from src.train_utils.wandb_utils import init_wandb


def resume_model_and_trainer(args, ckpt, world_size):
  model = get_model(args)
  model.load_state_dict(ckpt['model'])

  # Set up the datasets.
  train_dataloader = get_dataloader(args, split='train')
  valid_dataloader = get_dataloader(args, split='val')

  # Set up the train/eval functions.
  train_fn = partial(train_loop, args=args, loader=train_dataloader, model=model,
                     world_size=world_size)
  valid_fn = partial(eval_loop, args=args, loader=valid_dataloader, model=model,
                     world_size=world_size)
  logging_fn = [
                # partial(reconstruct_latents, dataset=args.dataset, loader=valid_dataloader, model=model, num_plots=6),
                partial(log_codebook_usage, dataset=args.dataset, loader=valid_dataloader, model=model,
                        batch_size=args.batch_size),
                ]
  tm = Train_Manager(args, train_fn=train_fn, valid_fn=valid_fn, logging_fn=logging_fn)
  return model, tm

def get_model_and_trainer(args, world_size):
  # Load the model. Note: using x GPUs will split the batch_size up x ways.
  model = get_model(args)

  # Set up the datasets.
  train_dataloader = get_dataloader(args, split='train')
  valid_dataloader = get_dataloader(args, split='val')

  # Set up the train/eval functions.
  train_fn = partial(train_loop, args=args, loader=train_dataloader, model=model,
                     world_size=world_size)
  valid_fn = partial(eval_loop, args=args, loader=valid_dataloader, model=model,
                     world_size=world_size)
  logging_fn = [
                # partial(reconstruct_latents, dataset=args.dataset, loader=valid_dataloader, model=model, num_plots=6),
                partial(log_codebook_usage, dataset=args.dataset, loader=valid_dataloader, model=model,
                        batch_size=args.batch_size),
                ]
  tm = Train_Manager(args, train_fn=train_fn, valid_fn=valid_fn, logging_fn=logging_fn)
  return model, tm

def train_model(world_size):
  args = train_parser()
  torch.manual_seed(args.seed)
  model, tm = get_model_and_trainer(args, world_size)
  tm.train(model)


if __name__ == '__main__':
  world_size = torch.cuda.device_count()  # Number of GPUs -- change w/ CUDA_VISIBLE_DEVICES env. variable.
  train_model(world_size)
