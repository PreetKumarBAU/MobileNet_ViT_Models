
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 
import torch.nn.functional as F

## IoULoss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

# DiceBCELoss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

## DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

## Focal Loss
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
        


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            #t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return 1 - intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS
    #class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def plot_image1(image, boxes, true_bboxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS
    #class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels )+ 10)]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Create a Rectangle patch
    for box in true_bboxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[-1],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)


    plt.show()





def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, y, l1 , a , b, c , v1, v2) in enumerate(tqdm(loader)):

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        a = a.to(config.DEVICE)
        b0 , b1 , b2 = (
            b[0].to(config.DEVICE),
            b[1].to(config.DEVICE),
            b[2].to(config.DEVICE),
        )


        with torch.no_grad():
            predictions =  model([x, a])

        batch_size = x.shape[0]

        ## Computing bboxes from cells predications, 
        bboxes = [[] for _ in range(batch_size)]

        ## 1. Scaling the Predications based on Anchor Sizes
        ## 2. Compute Predicated bboxes, by converting from cells to bboxes
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )

            ##
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        ## Compute Target bboxes from the target label
        true_bboxes = cells_to_bboxes(
            y[2], anchor, S=S, is_preds=False
        )
        '''
        true_bboxes1 = cells_to_bboxes(
            y[2], anchor, S=S, is_preds=False
        )
        
        true_bboxes2 = cells_to_bboxes(
            b[2], anchor, S=S, is_preds=False
        )
        true_bboxes = true_bboxes1 + true_bboxes2

        '''
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            ## Only taking target boxes which are greater than particular threshold
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y, l1 , a , b, c , v1, v2) in enumerate(tqdm(loader)):

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        a = a.to(config.DEVICE)
        b0 , b1 , b2 = (
            b[0].to(config.DEVICE),
            b[1].to(config.DEVICE),
            b[2].to(config.DEVICE),
        )

        with torch.no_grad():
            out = model([x, a])

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
        '''
        for i in range(3):
            b[i] = b[i].to(config.DEVICE)
            obj = b[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = b[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == b[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == b[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == b[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
        '''

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


train_image_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
train_label_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
train_flo_dir  =   r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
train_flowImg_dir  =  r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"

test_image_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
test_label_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
test_flo_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
test_flowImg_dir  = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"

val_image_dir  =  r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
val_label_dir  =  r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
val_flo_dir    =  r"C:\Users\azure.lavdierada\EM-Flow-Segmentation"
val_flowImg_dir = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation" 

def get_loaders(train_csv_path, test_csv_path, val_csv_path, BATCH_SIZE = 2, IMG_SIZE = 256):
    from dataset import ChagasDataset

    train_dataset = ChagasDataset(
        csv_file  = train_csv_path,
        img_dir   = train_image_dir,
        label_dir = train_label_dir,
        flo_dir =   train_flo_dir, 
        flo_img_dir= train_flowImg_dir,
        image_size=IMG_SIZE,
        transform=train_transforms,

    )

    test_dataset = ChagasDataset(
        csv_file  = test_csv_path,
        img_dir   = test_image_dir,
        label_dir = test_label_dir,
        flo_dir =   test_flo_dir, 
        flo_img_dir= test_flowImg_dir,
        image_size=IMG_SIZE,
        transform=test_transforms,
    )

    val_dataset = ChagasDataset(
        csv_file  = val_csv_path,
        img_dir   = val_image_dir,
        label_dir = val_label_dir,
        flo_dir =   val_flo_dir, 
        flo_img_dir= val_flowImg_dir,
        image_size=IMG_SIZE,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset= val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


    return train_loader, test_loader, val_loader


def get_loaders_UNet_AE(train_csv_path, test_csv_path, eval_csv_path):
    from dataset import UNet_AE_Dataset

    config = config_unet_ae
    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = UNet_AE_Dataset(
        train_csv_path,
        transform=config.train_transforms,
        img_dir=config.IMG_DIR,
        
        
    )
    test_dataset = UNet_AE_Dataset(
        test_csv_path,
        transform=config.test_transforms,
        img_dir=config.IMG_DIR,
        
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    eval_dataset = UNet_AE_Dataset(
        eval_csv_path,
        transform=config.train_transforms,
        img_dir=config.IMG_DIR,
        
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, eval_loader

from skimage.metrics import structural_similarity
from torch.autograd import Variable
import torch.nn as nn
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose(
    [ 
        #A.HorizontalFlip(p=0.5),
        #A.Blur(p=0.1),
        #A.CLAHE(p=0.1),
        #A.Posterize(p=0.1),
        #A.ToGray(p=0.1),
        #A.ChannelShuffle(p=0.05),
        #A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        #A.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
        ToTensorV2(transpose_mask=True, always_apply=True, p=1.0),
    ],
    additional_targets={'image1': 'image', 'image2': 'image' , 'image3': 'image' ,  'mask1': 'mask', 'mask2': 'mask',  'mask3': 'mask', 'flo1': 'image' ,  'flo2': 'image', 'flo3': 'image', 'flow_image1': 'image' ,  'flow_image2': 'image', 'flow_image3': 'image'},
                        
)



test_transforms = A.Compose(
    [ 

        #A.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
        ToTensorV2(transpose_mask=True, always_apply=True, p=1.0),
    ],
    additional_targets={'image1': 'image', 'image2': 'image' , 'image3': 'image' ,  'mask1': 'mask', 'mask2': 'mask',  'mask3': 'mask', 'flo1': 'image' ,  'flo2': 'image', 'flo3': 'image', 'flow_image1': 'image' ,  'flow_image2': 'image', 'flow_image3': 'image'},

                        
)

def save_predictions_as_imgs(
    loader, model,   folder="saved_images/", device="cuda" , 
):
    model.eval()
    with torch.no_grad():
        #unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        unorm = UnNormalize(mean=[0, 0, 0], std=[1,1, 1])
        
        for batch_idx, (image1, mask1, flow_image1, flo1, video_name1, img_name1 , image2, mask2, flow_image2, flo2, video_name2, img_name2 ,  image3, mask3, flow_image3, flo3, video_name3, img_name3) in enumerate(tqdm(random.sample(list(loader), 20))): 
            loss = 0
            #if batch_idx > 10:
            #    model.train()
            #    break

            img1 , img2 , img3 = image1.float().to(DEVICE) ,  image2.float().to(DEVICE),  image3.float().to(DEVICE)
            mask1 , mask2 , mask3 = mask1.float().to(DEVICE) , mask2.float().to(DEVICE) , mask3.float().to(DEVICE)
            flow_image1 , flow_image2 , flow_image3 = flow_image1.float().to(DEVICE) , flow_image2.float().to(DEVICE) , flow_image3.float().to(DEVICE)
            flo1 , flo2 , flo3 = flo1.float().to(DEVICE) , flo2.float().to(DEVICE) , flo3.float().to(DEVICE)
            
            frames = [img1 , img2 , img3]
            flows = [ flow_image1 , flow_image2 ]
            #frames_memory = [ frame.detach().clone() for frame in FRAMES] 
            frames_memory = [ frame.detach().clone().float().to(device=device) for frame in frames] 

            F = [ unorm(frame[0].detach().clone()) for frame in frames]
            F = [ frame.unsqueeze(dim=0) for frame in F]

            #frames_memory = FRAMES      ## It is a list with tensors
            bs = frames[0].shape[0]
            
            '''
            try:
                thresh.save(f"{folder}/thresh_{batch_idx}.png")
            except:
                cv2.imwrite(f"{folder}/thresh_{batch_idx}.png",thresh)
            '''

            losses = []
            #fake_image = torch.ones_like(frames_memory[0])
            #pred_image_given = False
            prediction  =  model( frames, flows )
            #prediction , _ , _  , _ , _ , _ , _ , _ , _= model(frames_memory )

            #pred_mean = torch.mean( prediction.detach(), dim = (1,2,3)  ).reshape(bs, 1)
            #pred_mean = Variable(pred_mean, requires_grad=False).float().to(device=device)

            ## Segmentation IOU of "pred1" and "mask1"
            #print(prediction.shape)
            #print(mask1.shape)
            loss1 = IoULoss()(prediction, mask1)
            losses.append(loss1)
            loss += loss1
            print(" loss::", loss)

            if len(prediction) > 1:
                #preds= [torch.sigmoid(pred) for pred in prediction]
                preds = [(pred > 0.5).float() for pred in prediction]
            ## Saving the Pred Frame
                torchvision.utils.save_image(
                    preds, f"{folder}/pred_{batch_idx}.png"
                )

            else:
                #preds= torch.sigmoid(prediction)
                preds = (prediction > 0.5).float()
            ## Saving the Pred Frame
                torchvision.utils.save_image(
                    preds, f"{folder}/pred____{batch_idx}.png"
                )
            
        ## Saving the Target Frame
            try:
                torchvision.utils.save_image(F[0].unsqueeze(1), f"{folder}/frame1_{batch_idx}.png")
                torchvision.utils.save_image(F[1].unsqueeze(1), f"{folder}/frame2_{batch_idx}.png")
                torchvision.utils.save_image(F[2].unsqueeze(1), f"{folder}/frame3_{batch_idx}.png")

            except:
                torchvision.utils.save_image(F[0], f"{folder}/frame1_{batch_idx}.png")
                torchvision.utils.save_image(F[1], f"{folder}/frame2_{batch_idx}.png")
                torchvision.utils.save_image(F[2], f"{folder}/frame3_{batch_idx}.png")

            ## Saving the Pred Frame
                torchvision.utils.save_image(
                    mask1, f"{folder}/mask1_{batch_idx}.png"
                )

    model.train()


def save_predictions_as_imgs1(
    loader, model,   folder="saved_images/", device="cuda" , 
):
    model.eval()
    with torch.no_grad():
        #unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        unorm = UnNormalize(mean=[0, 0, 0], std=[1,1, 1])
        #for batch_idx, FRAMES in enumerate(tqdm(random.sample(loader, 20))): 
        for batch_idx, FRAMES in enumerate(tqdm(loader)): 
            if batch_idx > 10:
                model.train()
                break

            #frames_memory = [ frame.detach().clone() for frame in FRAMES] 
            frames_memory = [ frame.detach().clone().float().to(device=device) for frame in FRAMES] 


            F = [ unorm(frame[0].detach().clone()) for frame in FRAMES]
            F = [ frame.unsqueeze(dim=0) for frame in F]

            #frames_memory = FRAMES      ## It is a list with tensors
            bs = FRAMES[0].shape[0]
            
            
            SSIM_list = []
            for j in range( len(FRAMES) - 1 ):
                for i in range( bs ):
                    before = F[j][i].detach().cpu().permute( 1, 2, 0).numpy()
                    after = F[j+1][i].detach().cpu().permute( 1, 2, 0).numpy()

                    # Convert images to grayscale
                    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
                    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

                    score, diff = structural_similarity(before_gray, after_gray,  full=True , gradient= False)
                    
                    SSIM_list.append(score)
            
            diff = (diff * 255).astype("uint8")

            ## Thresholding
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            mask = np.zeros((before.shape[0], before.shape[1] ), dtype='uint8')

            print('Len of Contours::',len(contours))
            for c in contours:
                area = cv2.contourArea(c)
                if area > 10:
                    
                    #x,y,w,h = cv2.boundingRect(c)
                    cv2.drawContours(mask, [c], 0, 255, -1)  ## img, contour,  color, thickness
                    
            try:
                thresh.save(f"{folder}/thresh_{batch_idx}.png")
            except:
                cv2.imwrite(f"{folder}/thresh_{batch_idx}.png",thresh)

            target_frame = torch.from_numpy(diff).float().to(device=device)

            SSIM_tensor = torch.from_numpy( np.asarray(SSIM_list) ).unsqueeze(1).float().to(device=device) ## 


            losses = []
            fake_image = torch.ones_like(frames_memory[0])
            pred_image_given = False
            prediction , _ , _ , _ , _, _ , _ , _, _ =  model( frames_memory, epoch = 0 )
            #prediction , _ , _  , _ , _ , _ , _ , _ , _= model(frames_memory )

            pred_mean = torch.mean( prediction.detach(), dim = (1,2,3)  ).reshape(bs, 1)
            pred_mean = Variable(pred_mean, requires_grad=False).float().to(device=device)

            loss = nn.MSELoss( )( pred_mean , (1- SSIM_tensor))
            #loss = nn.MSELoss( )( pred_mean , SSIM_tensor)
            print(" loss::", loss)

            if len(prediction) > 1:
                #preds= [torch.sigmoid(pred) for pred in prediction]
                preds = [(pred > 0.5).float() for pred in prediction]
            ## Saving the Pred Frame
                torchvision.utils.save_image(
                    preds, f"{folder}/pred_{batch_idx}.png"
                )

            else:
                #preds= torch.sigmoid(prediction)
                preds = (prediction > 0.5).float()
            ## Saving the Pred Frame
                torchvision.utils.save_image(
                    preds, f"{folder}/pred____{batch_idx}.png"
                )
            
        ## Saving the Target Frame
            try:
                torchvision.utils.save_image(F[0].unsqueeze(1), f"{folder}/frame1_{batch_idx}.png")
                torchvision.utils.save_image(F[1].unsqueeze(1), f"{folder}/frame2_{batch_idx}.png")

            except:
                torchvision.utils.save_image(F[0], f"{folder}/frame1_{batch_idx}.png")
                torchvision.utils.save_image(F[1], f"{folder}/frame2_{batch_idx}.png")

            
    model.train()


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    
    #y[0] is torch.Size([32, 3, 13, 13, 6])
    #y[1] torch.Size([32, 3, 26, 26, 6])
    #y[2] torch.Size([32, 3, 52, 52, 6])
    
    x, y, l1 , a , b, c , v1, v2  = next(iter(loader))
    x = x.to("cuda")
    a = a.to("cuda")
    with torch.no_grad():
        out = model([x, a])
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        true_bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = y[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                y[i], anchor, S=S, is_preds=False
            )
            for idx, (box) in enumerate(boxes_scale_i):
                true_bboxes[idx] += box
            '''
            for idx, (box) in enumerate(boxes_scale_i):
                true_bboxes[idx] += box
            batch_size, A, S, _, _ = b[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                b[i], anchor, S=S, is_preds=False
            )
            
            
            for idx, (box) in enumerate(boxes_scale_i):
                true_bboxes[idx] += box
            '''
        #

    for i in range(batch_size):
        #print(y[i].shape)
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        nms_boxes1 = non_max_suppression(
             true_bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        #try:
        plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes, nms_boxes1)
        #except:
        #    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

    model.train()

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            #t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

