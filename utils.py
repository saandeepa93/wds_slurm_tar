import random 
import os 
import numpy as np 
import torch 
import cv2
import argparse
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve
from scipy import interpolate
import re
import pickle

from transforms import *
from torchvision import transforms
from torchvision.utils import save_image

def load_pickle(subject, root):
  fpath = os.path.join(root, f"{subject}.pickle")
  with open(fpath, 'rb') as inp:
    pickl_file = pickle.load(inp)
  return pickl_file


class DataAugmentationForVideoMAE(object):
  def __init__(self):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((112, 112))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        # normalize,
    ])

  def __call__(self, images):
    process_data = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    # repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr

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


def save_frames(frames, path):
  frames = frames.permute(1, 0, 2, 3).cpu().detach()
  for i, frame in enumerate(frames):
    # Save each frame to a file
    # The file name will be "frame_{i}.png", where {i} is the frame index
    save_image(frame, f'{path}/frame_{i}.png')

def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
    # DDP configs:
  parser.add_argument('--world-size', default=-1, type=int, 
                      help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int, 
                      help='node rank for distributed training')
  parser.add_argument('--dist-url', default='env://', type=str, 
                      help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str, 
                      help='distributed backend')
  parser.add_argument('--local_rank', default=-1, type=int, 
                      help='local rank for distributed training')
  parser.add_argument('--gpu', default=-1, type=int, 
                      help='local rank for distributed training')
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
  arr = arr.permute(1, 2, 3, 0)
  # arr = arr.permute(0, 2, 3, 1)
  arr = arr.detach().cpu().numpy()
  b, h, w, c = arr.shape
  for i in range(b):
    img = arr[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    expression = 0 #exp[i].item()
    # if expression != 2:
    #   continue
    img = cv2.putText(img, str(expression), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)
    # cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{i}_{mode}.png"), img)
    cv2.imwrite(f"./data/vid_loader/test/img_{i}_{mode}.png", img)


def compute_ROC(labels, scores):
  np_scores = np.array(scores)
  y_preds = np.where(np_scores > 0.5, 1, 0)

  fpr, tpr, thresholds = roc_curve(labels, scores)
  fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  f_interp = interpolate.interp1d(fpr, tpr)
  tpr_at_fpr = [f_interp(x) for x in fpr_levels]
  for (far, tar) in zip(fpr_levels, tpr_at_fpr):
      print('TAR @ FAR = {} : {}'.format(far, tar))
  acc = accuracy_score(y_preds, labels)
  return acc

def viz_wts(arr, exp, cfg, mode):

  arr = arr.permute(1, 2, 3, 0)
  arr = arr.detach().cpu().numpy()
  b, h, w, c = arr.shape
  
  for i in range(b):
    img = arr[i]
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    expression = exp[i].item()
    img = cv2.putText(img, str(expression), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)

    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{i}_{expression}.png"), img)
