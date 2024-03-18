from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.ROOT = "/dataset/DISFA/Videos_LeftCamera"
_C.PATHS.PROOT = "/dataset/DISFA/Videos_LeftCamera"
_C.PATHS.TARROOT = "/dataset/DISFA/Videos_LeftCamera"
_C.PATHS.OFROOT = "/dataset/DISFA/Videos_LeftCamera"


_C.DATASET = CN()
_C.DATASET.IMG_SIZE=224
_C.DATASET.CLIP_LGT = 4
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.WDS_CHUNKS = 1
_C.DATASET.SPLIT = ""


_C.TRAINING = CN()
_C.TRAINING.ITER=100
_C.TRAINING.LR=1e-4
_C.TRAINING.WT_DECAY=1e-5
_C.TRAINING.BATCH_SIZE=256
_C.TRAINING.DISTRIBUTED=False
_C.TRAINING.FEATS=0
_C.TRAINING.MODEL="ELASTIC"


_C.TEST = CN()
_C.TEST.FOLD = 1

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()