import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import random
import torchvision

#from models.UNet_Structural_Similiarity import Build_UNet
from MobileVIT.mobilev2_vit_unet import MobileNetV2_UNet_Videos
from train_mobileVitUNet import  train_fn

from utils import (
    load_checkpoint,
    save_checkpoint,
    save_predictions_as_imgs,
    UnNormalize,
)
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples, intersection_over_union, IoULoss, DiceBCELoss, DiceLoss, FocalLoss
)

import time
import shutil
from tqdm import tqdm
import cv2
import numpy as np

import os
# Hyperparameters etc.
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
#TRAIN_IMG_DIR = "ChagasVideos/train_images/"
#TRAIN_MASK_DIR = "ChagasVideos/train_masks/"
#VAL_IMG_DIR = "ChagasVideos/val_images/"
#VAL_MASK_DIR = "ChagasVideos/val_masks/"
NUM_FRAMES = 2   ## 2 
model_train_state = True


# MobileNetV2_UNet_Videos Model Configurations 
vit_size = "xxs"
width_mult=1
IMG_SIZE = 256
patch_size = 2

import torch
import torch.optim as optim
import math
#from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples, 
    intersection_over_union,
    seed_everything
)


def main():

    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)   # C:\Users\azure.lavdierada\TPS\MobileVIT

    seed_everything()  # If you want deterministic behavior

    train_loader, test_loader, val_loader = get_loaders(
        train_csv_path=  dir_path + "\\train.csv", test_csv_path= dir_path + "\\test.csv", val_csv_path= dir_path + "\\val.csv", 
        BATCH_SIZE= TRAIN_BATCH_SIZE, IMG_SIZE= IMG_SIZE
    )

    print("train_loader::", len(train_loader))
    print("val_loader::", len(val_loader))
    print("test_loader::", len(test_loader))

    model = MobileNetV2_UNet_Videos(vit_size , out_ch=1, input_size=IMG_SIZE, patch_size = 2, width_mult=1.) 

    #of_model.load_state_dict(torch.load(ckpt_name))
        
    model.to(DEVICE)

    if model_train_state:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        model.eval()

    ## Loss Function Defined
    #loss_fn = nn.MSELoss()
    loss_fn = IoULoss()
    

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model )
        save_predictions_as_imgs(
                test_loader, model, folder="saved_images/", device=DEVICE
            )

    
    if not LOAD_MODEL:
        #NUM_EPOCHS = 1
        for epoch in range(NUM_EPOCHS):

            '''
            if epoch ==0:
                save_predictions_as_imgs(
                    test_loader , model, folder="saved_images/", device=DEVICE
                )
            '''

            train_fn(train_loader, model, optimizer, loss_fn, scaler , epoch )

            #train_fn(shuffled_train_frames_list, model, optimizer, loss_fn, scaler , epoch )
            
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            
            #if model_train_state:
            #    save_checkpoint(model )
            
            # check accuracy
            #for shuffled_val_frames in shuffled_val_frames_list:
            #    check_accuracy(shuffled_val_frames, model, device=DEVICE)

            # print some examples to a folder

            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )
            
            
            save_predictions_as_imgs(
                test_loader, model, folder="results_images/", device=DEVICE
            )
            

if __name__ == "__main__":

    main()


