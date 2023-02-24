import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2

class CellsImage(Dataset):
    def __init__(self, data_list, transform=None, train=False):
        super(CellsImage, self).__init__()
        self.data_list = data_list
        self.nSamples = len(data_list)
        self.transform = transform
        self.train = train
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        Img_path = self.data_list[index]
        fname = Img_path
        gt_path = fname.replace("jpg", "txt").replace("images_crop_plus", "gt_density_map_plus")
        
        with open(gt_path, "r") as f:
            gt_count = int(float(f.readlines()[0]))
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            transformed_img = self.transform(img)
        return fname, transformed_img, gt_count

class CellsImage_no_gt(Dataset):
    def __init__(self, data_list, transform=None, train=False):
        super(CellsImage_no_gt, self).__init__()
        self.data_list = data_list
        self.nSamples = len(data_list)
        self.transform = transform
        self.train = train
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        Img_path = self.data_list[index]
        fname = Img_path
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            transformed_img = self.transform(img)
        return fname, transformed_img