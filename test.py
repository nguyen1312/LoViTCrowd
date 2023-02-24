import argparse
import torch
from src.dataset_test import CellsImage
import glob2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import torch.nn as nn
from torchvision import transforms
import math
import torch
import os
import logging
import numpy as np
from fastprogress import progress_bar
import scipy.io as io


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inference number of people in a scene')
    parser.add_argument('--ckpt', type=str, default="pretrained_weight/shanghaiB/best_weights_shanghaiB_1024x768.pth")
    parser.add_argument('--datatest-path', type=str, default="datatest/shB_original/test/images_crop_plus")
    parser.add_argument('--datatest-name', type=str, default="shanghaiB")
    parser.add_argument('--batch-size', type=int, default=768)
    
    args = parser.parse_args()    
    DEVICE = "cuda:2" if torch.cuda.is_available() == True else "cpu"
    CKPT_PATH = args.ckpt
    DATASET_NAME = args.datatest_name
    DATASET_PATH = args.datatest_path
    BATCH_SIZE = args.batch_size
    
    from models.LoViTCrowd import *
    model = Net()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    list_key = list(ckpt.keys())
    for key in list_key:    
        new_key = key.replace("module.", "")
        ckpt[new_key] = ckpt[key]
        del ckpt[key]
    list_key = list(ckpt.keys())
    for key in list_key:
        if key.find("part_cnn.patch_embed.backbone.7.2.3.weight") != -1:
            new_key = key.replace("7.2.3", "7.2.5")
            ckpt[new_key] = ckpt[key]
            del ckpt[key]
    try:
        model.load_state_dict(ckpt)
    except:
        list_key = list(ckpt.keys())
        for key in list_key:
            if key.find("part_cnn.patch_embed.backbone.7.2.5.weight") != -1:
                new_key = key.replace("7.2.5", "7.2.3")
                ckpt[new_key] = ckpt[key]
                del ckpt[key]
        model.load_state_dict(ckpt)
        
        
    model = model.to(DEVICE)
    model.eval()
    print("Load pretrained weight successfully!")
    
    
    test_list = glob2.glob(f"{DATASET_PATH}/*.jpg")
    test_list = sorted(test_list, key=lambda x: (int(x.split("/")[-1].replace(".jpg", "").split("_")[-1]), int(x.split("/")[-1].replace(".jpg", "").split("_")[0])))
    test_dataset = CellsImage(
                            test_list,
                            transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE, 
        drop_last=False
    )
    
    
    mae = 0.0
    mse = 0.0
    visi = []
    index = 0
    preds = list()
    gts = list()
    couple_list = list()
    for i_, (fname, img, gt_count) in enumerate(test_loader):

        fnames = list(map(lambda x: "_".join(x.split("/")[-1].replace(".jpg", "").split("_")[-2:]), fname))
        img = img.to(DEVICE)
        with torch.no_grad():
            assert len(set(fnames)) == 1
            out1, _ = model(img)
        assert DATASET_NAME in ["shanghaiA", "shanghaiB", "ucf-qnrf", "mall"], "Invalid"
        if DATASET_NAME == "shanghaiA":
            ground_truth = io.loadmat(f"data/part_A_final/test_data/ground_truth/GT_{fnames[0]}.mat")
            gt_count_ref = ground_truth["image_info"][0][0][0][0][1][0][0]
        elif DATASET_NAME == "shanghaiB":
            ground_truth = io.loadmat(f"data/part_B_final/test_data/ground_truth/GT_{fnames[0]}.mat")
            gt_count_ref = ground_truth["image_info"][0][0][0][0][1][0][0]
        elif DATASET_NAME == "mall":
            import pickle
            with open("data/data_mall/mall_pad_gt.pkl", "rb") as f:
                gt_counts = pickle.load(f)    
            ids = int(fnames[0].split("/")[-1].replace(".jpg", "").split("_")[-1])
            gt_count_ref = len(gt_counts[ids-1])
        elif DATASET_NAME == "ucf-qnrf":
            name_frame = fnames[0]
            id_frame = name_frame.split("_")[-1].zfill(4)
            updated_name_frame = f"img_{id_frame}"
            ground_truth = io.loadmat(f"data/UCF-QNRF/UCF-QNRF_ECCV18/Test/{updated_name_frame}_ann.mat")
            Gt_data = ground_truth['annPoints']
            gt_count_ref = Gt_data.shape[0]

        couple_list.append((out1.detach().cpu(), gt_count_ref))
        mask = torch.tensor(out1 >= 0.0, dtype=torch.float32)
        out1 = out1 * mask
        pred_count = torch.sum(out1).item() 
        gt_count = gt_count.type(torch.FloatTensor).to(DEVICE).unsqueeze(1)
        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count_ref - pred_count)
        mse += abs(gt_count_ref - pred_count) * abs(gt_count_ref - pred_count)
        if i_ % 1 == 0:
            gts.append(gt_count_ref)
            preds.append(pred_count)
            print('{fname} Gt_ref {gt_ref:.2f} Gt {gt:.2f} Pred {pred} MAE: {cur_mae}'.format(fname=fnames[0], gt_ref=gt_count_ref, gt=gt_count, pred=pred_count, cur_mae=mae/(i_+1)))
    len_data = len(test_list)/BATCH_SIZE
    mae = mae * 1.0 / (len_data)
    mse = math.sqrt(mse / (len_data))
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))