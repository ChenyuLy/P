import torch
from PIL import Image

from torch.utils.data import Dataset
from .base_dataset import BaseDataset
from .builder import DATASETS
import os
import numpy as np


@DATASETS.register_module()
class Flower_photos(BaseDataset):
    def load_annotations(self):
        data_infos = []
        CLASSES = os.listdir(self.data_prefix)

        for classdir in CLASSES:
            samples = os.listdir(self.data_prefix + classdir)
            clas = [CLASSES.index(classdir)] * len(samples)
            samples = zip(samples, clas)
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': classdir + '/'+filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)

        return data_infos
