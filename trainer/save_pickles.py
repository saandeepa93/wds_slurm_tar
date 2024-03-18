import webdataset as wds

from imports import * 


if __name__ == "__main__":
  root_dir = "/data/scanavan/BP4D+/2D+3D"
  
  subject_dict = {}
  for subject_entry in os.scandir(root_dir):
    if not subject_entry.is_dir():
      continue
    subject = subject_entry.name 
    subject_dir = subject_entry.path 
    subject_dict[subject] = {}

    for task_entry in os.scandir(subject_dir):
      if task_entry.name not in [f"T{k}" for k in range(1, 9)]:
        continue
      if not task_entry.is_dir():
        continue
      task = task_entry.name 
      task_dir = task_entry.path
      subject_dict[subject][task] = []

      img_lst = []
      for img_entry in os.scandir(task_dir):
        if img_entry.name.split('.')[-1] != "jpg":
          continue
        img = img_entry.name 
        img_path = img_entry.path 
        img_lst.append(img_path)

      with open(f'./data/BP4D+/pickles/{subject}_{task}.pickle', 'wb') as f:
        pickle.dump(img_lst, f, pickle.HIGHEST_PROTOCOL)


