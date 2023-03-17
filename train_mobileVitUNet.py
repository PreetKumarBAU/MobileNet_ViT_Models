import torch
import torch.optim as optim
import math
#from model import YOLOv3
from MobileVIT.mobilev2_vit_unet import MobileNetV2_UNet_Videos


from tqdm import tqdm
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

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torchvision


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    nan_values =0
    outputs= []
    video_names = []
    n = 0
    for batch_idx, (image1, mask1, flow_image1, flo1, video_name1, img_name1 , image2, mask2, flow_image2, flo2, video_name2, img_name2 ,  image3, mask3, flow_image3, flo3, video_name3, img_name3)   in enumerate(loop):
    #for batch_idx, (x, y, l1 , a , b, c , v1, v2) in enumerate(loop):

        #print(type(img1))
        img1 , img2 , img3 = image1.float().to(DEVICE) ,  image2.float().to(DEVICE),  image3.float().to(DEVICE)
        mask1 , mask2 , mask3 = mask1.float().to(DEVICE) , mask2.float().to(DEVICE) , mask3.float().to(DEVICE)
        flow_image1 , flow_image2 , flow_image3 = flow_image1.float().to(DEVICE) , flow_image2.float().to(DEVICE) , flow_image3.float().to(DEVICE)
        flo1 , flo2 , flo3 = flo1.float().to(DEVICE) , flo2.float().to(DEVICE) , flo3.float().to(DEVICE)

        frames = [img1 , img2 , img3]
        flows = [ flow_image1 , flow_image2 ]



        #print(" y0, y1, y2::", y0, y1, y2)
        with torch.cuda.amp.autocast():
            loss = 0
            output = model( frames, flows)

        ## Segmentation IOU of "pred1" and "mask1"
        loss1 = IoULoss()(output, mask1)
        loss += loss1

        '''
        ## IOU b/w Pred1 and Pred2
        iou_loss_Pred1_Pred2 = IoULoss()(output[0], output[1])
        loss += iou_loss_Pred1_Pred2

        ## IOU b/w Pred2 and Pred3
        iou_loss_Pred2_Pred3 = IoULoss()(output[1], output[2])
        loss += iou_loss_Pred2_Pred3
        '''

        print("Overall Loss::", loss)
        if math.isnan(loss):
            nan_values += 1
        else:
            
            losses.append(loss.item())
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)
    print("Number of Nan Losses:", nan_values)


