from imports import * 

import torch 
from torch.utils.data import Dataset 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

import tarfile

from face_alignment import mtcnn
from face_detection import RetinaFace
from transforms import *
from imports import *

class DataAugmentationForVideoMAE(object):
  def __init__(self, cfg):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
        # normalize,
    ])

  def __call__(self, images):
    process_data = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr



class BP4D(Dataset):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg 
    self.transform = DataAugmentationForVideoMAE(cfg)
    self.all_files = glob.glob(os.path.join(cfg.PATHS.PROOT, "*.pickle"))
    self.face_detector_backup = RetinaFace()
    self.face_detector = mtcnn.MTCNN(device='cpu', crop_size=(cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE))
  
  def _loadFrames(self, fpaths):
    cropped_images = []
    for fpath in fpaths:
      image = Image.open(fpath).convert('RGB')
      img_arr = np.array(image)
      faces = self.face_detector_backup(img_arr)
      if len(faces) == 0:
        crop_img = img_arr
      else:
        bboxes = faces[0][0]
        bboxes = bboxes.astype(np.int32)
        x1, y1, x2, y2 = bboxes.data
        x1 = max(x1, 0)
        x2 = max(x2, 0)
        y1 = max(y1, 0)
        y2 = max(y2, 0)
        crop_img = img_arr[y1:y2, x1:x2, :]
      img = Image.fromarray(crop_img).convert('RGB')
      cropped_images.append(img)
    return cropped_images

    
  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    subject_task_path = self.all_files[idx]
    subject_task = subject_task_path.split('/')[-1].split('.')[0]
    with open(subject_task_path, 'rb') as inp:
      fpaths = pickle.load(inp)
    images = self._loadFrames(fpaths)
    process_data = self.transform(images)
    process_data = process_data.contiguous().view((len(fpaths), 3) + process_data.size()[-2:]).transpose(0,1)

    return process_data, subject_task

class BP4DTars(Dataset):
  def __init__(self, cfg, mode):
    super().__init__()
    self.cfg = cfg

    return self._getAllFiles()
  
  def _getAllFiles(self):
    tar_urls = []
    openface_dict = {}
    tar_files = glob.glob(os.path.join(self.cfg.PATHS.TARROOT, "*.tar"))
    for tar_path in tar_files:
      archive = tarfile.open(tar_path, 'r')
      for member in archive.getmembers():
        if "meta" in member.name:
          meta_file_obj = archive.extractfile(member)
          meta_content = meta_file_obj.read().decode('utf-8')
          df = pd.read_csv(os.path.join(self.cfg.PATHS.OFROOT, f"{meta_content}", f"{meta_content}.csv"))
          openface_dict[meta_content] = df

    return openface_dict
  






















