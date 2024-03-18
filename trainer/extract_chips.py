import sys
sys.path.append('.')

import torch 
from torch.utils.data import DataLoader
from torch import nn 
import webdataset as wds
import io

from imports import * 
from utils import *
from dataset import BP4D
from configs import get_cfg_defaults

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
  dataset = BP4D(cfg)
  loader = DataLoader(dataset, batch_size=1, shuffle=False)

  output_shard_pattern = f"{cfg.PATHS.TARROOT}/shard-%06d.tar"
  sink =  wds.ShardWriter(output_shard_pattern, maxcount=1, maxsize=10e9, verbose=0)

  pbar = tqdm(loader, desc=f"Creating tars...")
  for b, (video, subject_task) in enumerate(pbar):
    buffer = io.BytesIO()
    torch.save(video.contiguous().cpu().detach(), buffer)  # Move tensor to CPU before serialization
    video_bytes = buffer.getvalue()

    subject_task_bytes = subject_task[0].encode("utf-8")
    pbar.set_description(f"Subject: {subject_task[0]}")

    sample = {  
      "__key__": f"sample{b:06d}",
      "video.pt": video_bytes,
      "meta": subject_task_bytes
    }
    sink.write(sample)
  
  sink.close()
