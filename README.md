 # **Training with Webdataset under distributed setting**

 ## This repo contains code to handle large dataset and perform efficient IO operations to speed up the training. This code is written for [BP4D](https://paperswithcode.com/dataset/bp4d) and [BP4D+](https://www.sciencedirect.com/science/article/pii/S0262885614001012) dataset but can be extended to any dataset.

 * `./trainer/save_pickles.py` Contains code to save all the filepaths per subject and task. Can be grouped however you want.
 * `./trainer/extract_chips.py` Contains code to perform face detection for each group and save it into a `.tar` format. It also saves the meta information such as filename, class, subject_name and task.
 * `./trainer/extract_openface.py` Extracts [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) features by loading the tar files. Uses my external library [TorchOpenFace](https://github.com/saandeepa93/TorchOpenFace). See repo README and wiki for build details.
 * `./trainer/train_wds` Trains a model using webdataset library for IO loading under distributed setting. Uses SLURM based training and Torch DDP.