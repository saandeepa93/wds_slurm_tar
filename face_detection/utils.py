import random 
import os 
import numpy as np 
import torch 
import cv2
import argparse
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)


def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
      # distributed training parameters
  # parser.add_argument('--local_rank', default=-1, type=int)
  # parser.add_argument('--world_size', default=1, type=int,
                      # help='number of distributed processes')
  # parser.add_argument('--dist_on_itp', action='store_true')
  # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
  args = parser.parse_args()
  return args

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True


def get_metrics(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  mcc = matthews_corrcoef(y_true, y_pred)
  conf = confusion_matrix(y_true, y_pred)
  return round(acc, 3), round(mcc, 2), conf


class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

def plot_loader_imgs(arr, exp, cfg, mode):
  arr = arr.permute(0, 2, 3, 1)
  arr = arr.detach().cpu().numpy()
  b, h, w, c = arr.shape
  for i in range(b):
    img = arr[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    expression = exp[i].item()
    if expression != 2:
      continue
    img = cv2.putText(img, str(expression), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)

    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{i}_{mode}.png"), img)