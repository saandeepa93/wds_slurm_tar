import sys
sys.path.append('.')

import io
import braceexpand

import torch 
from torch.utils.data import DataLoader
from torch import nn 

import webdataset as wds
from einops import rearrange

# DISTRIBUTED COMPUTING
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group
from torchvision.transforms import ToPILImage
import torch.distributed as dist


from imports import * 
from utils import *
from dataset import BP4D
from configs import get_cfg_defaults

def ddp_setup(args):
  if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
  args.distributed = args.world_size > 1
  ngpus_per_node = torch.cuda.device_count()
  if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
      args.rank = args.local_rank
      args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      args.rank = int(os.environ['SLURM_PROCID'])
      args.gpu = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    if args.rank!=0:
      def print_pass(*args,  **kwargs):
          pass
      builtins.print = print_pass

    torch.cuda.set_device(args.gpu)

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


def preprocess(batch):
  video = batch['video.pt']
  buffer1 = io.BytesIO(video)
  tensor1 = torch.load(buffer1)

  meta = batch['meta']
  meta = meta.decode('utf-8')
  
  fnames = [f"{k}.jpg" for k in range(tensor1.size(2))]

  batch['video.pt'] = tensor1
  batch['meta'] = [meta]
  batch['fnames'] = [fnames]
  return batch

def prepare_loader(cfg, args):

  train_url = "./data/BP4D/tars/shard-{000000..000324}.tar"
  val_url = "./data/BP4D/tars/shard-{000011..000015}.tar"


  if cfg.TRAINING.DISTRIBUTED:
    world_size =  dist.get_world_size()
    
    train_urls = list(braceexpand.braceexpand(train_url))
    val_urls = list(braceexpand.braceexpand(val_url))

    train_ds_size = len(train_urls) * cfg.DATASET.WDS_CHUNKS
    val_ds_size = len(val_urls) * cfg.WDS_CHUNKS.WDS_CHUNKS

    train_dataset = wds.WebDataset(train_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess)\
      .to_tuple("video.pt", "meta")
      
    val_dataset = wds.WebDataset(val_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess)\
      .to_tuple("video.pt", "meta")
    

    
    train_n_batches =  max(1, train_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    train_loader = train_loader.unbatched().shuffle(1000).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(train_n_batches)

    val_n_batches =  max(1, val_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)
    val_loader = val_loader.unbatched().shuffle(0).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(val_n_batches)

  else:
    train_dataset = wds.WebDataset(train_url)
    val_dataset = wds.WebDataset(val_url)

    train_dataset = train_dataset\
            .map(preprocess)\
            .to_tuple("video.pt", "meta", "fnames")\
            # .shuffle(5000)\
            # .batched(cfg.TRAINING.BATCH_SIZE, partial=False)
    val_dataset = val_dataset\
              .map(preprocess)\
              .to_tuple("video.pt", "meta", "fnames")\
              # .shuffle(0)\
              # .batched(cfg.TRAINING.BATCH_SIZE, partial=False) 

    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)
    train_loader = train_loader.unbatched().shuffle(0).batched(cfg.TRAINING.BATCH_SIZE)
    
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = val_loader.unbatched().shuffle(0).batched(cfg.TRAINING.BATCH_SIZE)

  return train_loader, val_loader

def intialize_openface():
  root_dir = "/data/saandeepaath/TorchOpenFace"
  model_dir = os.path.join(root_dir, "models")
  dest_dir = "./data/BP4D/openface"
  lib_path = os.path.join(root_dir, "./build/lib/TorchFace/libTorchFace.so")

  misc_args = {
    "vis": True, 
    "rec": True,
    "first_only": True,
    "gallery_mode": True
  }
  torch.classes.load_library(lib_path)
  openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
  openface = torch.classes.TorchFaceAnalysis.TorchFaceLandmarkImg(openface_args, misc_args)
  return openface

def extract_openface(cfg, train_loader):
  openface = intialize_openface()

  if cfg.TRAINING.DISTRIBUTED:
    is_master = args.rank == 0
    pbar = tqdm(range(cfg.TRAINING.ITER), disable=not is_master)
  else:
    pbar = tqdm(range(cfg.TRAINING.ITER))
  
  pbar = tqdm(train_loader, desc=f"extracting OpenFace...")
  for b, (video, meta, fnames) in  enumerate(pbar):
    video = rearrange(video, 'b c t h w -> (b t) c h w')
    ex_args = {
      "fname": meta,
      "frame_lst": fnames[0]
    }
    pbar.set_description(f"Subject_Task: {meta[0]}")
    openface.ExtractFeatures(video.clone().detach().cpu().contiguous(), ex_args)

if __name__ == "__main__":
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  torch.cuda.empty_cache()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/", f"{args.config}.yaml")

  ckp_path = f"./checkpoints/{args.config}"
  mkdir(ckp_path)

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()

  if cfg.TRAINING.DISTRIBUTED:
    ddp_setup(args)
  
  train_loader, val_loader = prepare_loader(cfg, args)
  extract_openface(cfg, train_loader)




