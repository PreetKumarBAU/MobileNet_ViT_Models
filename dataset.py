"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def inbetween(minv, val, maxv):
    return float(min(maxv, max(minv, val))) 
import cv2

## return frame1, frame2, frame3, mask1, mask2, mask3, flow_img1, flow_img2, flow1, flow2, video_name , image_name1, image_name2, image_name3

class ChagasDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        flo_dir, 
        flo_img_dir,
        image_size=256,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)   ## It contains the "frameName" and "label txt file name"
        self.img_dir = img_dir
        self.label_dir = label_dir               ## It contains .txt files for each frame, having 
        self.flo_img_dir = flo_img_dir
        self.flo_dir = flo_dir
        self.image_size = image_size
        self.transform = transform


    def __len__(self):
        return len(self.annotations)
        #return len(self.annotations)//3 + len(self.annotations)%3


    def read_flow(self,filename):
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Invalid .flo file')
                return None
            else:
                w = int(np.fromfile(f, np.int32, count=1))
                h = int(np.fromfile(f, np.int32, count=1))
                flow = np.fromfile(f, np.float32, count=2*w*h)
                flow = np.resize(flow, (h, w, 2))
                return flow


    ## csv file has flow, image , gt_mask , flow_img
    def __getitem__(self, index):
        '''
        if index> 0 :
            index += 2

    
        print("index:",index)
        '''
        try:
            flow_path1 = os.path.join(self.flo_dir , self.annotations.iloc[index, 0])
            img_path1 = os.path.join(self.img_dir , self.annotations.iloc[index, 1])
            label_path1 = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
            flow_image_path1 = os.path.join(self.flo_img_dir, self.annotations.iloc[index, 3])


            video_name1 = self.annotations.iloc[index, 0].split("/")[-2]
            img_name1 = self.annotations.iloc[index, 0].split("/")[-1].split(".")[0]
            ## Image
            image1 = np.array(Image.open(img_path1).convert("RGB"))
            image1 = cv2.resize(image1, (self.image_size, self.image_size))
            ## Mask
            mask1 = np.array(Image.open(label_path1).convert("L"))
            mask1 = cv2.resize(mask1, (self.image_size, self.image_size))
            ## FlowImage
            flow_image1 = np.array(Image.open(flow_image_path1).convert("RGB"))
            flow_image1 = cv2.resize(flow_image1, (self.image_size, self.image_size))
            ## Flo 
            flo1 = self.read_flow( flow_path1)
            flo1 = np.resize(flo1, (self.image_size, self.image_size , 2))


        ## For Image2 and Label2 from "index + 1 "
            flow_path2 = os.path.join(self.flo_dir , self.annotations.iloc[index+1, 0])
            img_path2 = os.path.join(self.img_dir , self.annotations.iloc[index+1, 1])
            label_path2 = os.path.join(self.label_dir, self.annotations.iloc[index+1, 2])
            flow_image_path2 = os.path.join(self.flo_img_dir, self.annotations.iloc[index+1, 3])
            ## VideoName2 and image_name2
            video_name2 = self.annotations.iloc[index+1, 0].split("/")[-2]
            img_name2 = self.annotations.iloc[index + 1 , 0].split("/")[-1].split(".")[0]
            ## Image2
            image2 = np.array(Image.open(img_path2).convert("RGB"))
            image2 = cv2.resize(image2, (self.image_size, self.image_size))
            ## Mask2
            mask2 = np.array(Image.open(label_path2).convert("L"))
            mask2 = cv2.resize(mask2, (self.image_size, self.image_size))
            ## FlowImage2
            flow_image2 = np.array(Image.open(flow_image_path2).convert("RGB"))
            flow_image2 = cv2.resize(flow_image2, (self.image_size, self.image_size))
            ## Flo2
            flo2 = self.read_flow(flow_path2)
            flo2 = np.resize(flo2, (self.image_size, self.image_size , 2))


        ## For Image3 and Label3 from "index + 2 "
            flow_path3 = os.path.join(self.flo_dir , self.annotations.iloc[index+2, 0])
            img_path3 = os.path.join(self.img_dir , self.annotations.iloc[index+2, 1])
            label_path3 = os.path.join(self.label_dir, self.annotations.iloc[index+2, 2])
            flow_image_path3 = os.path.join(self.flo_img_dir, self.annotations.iloc[index+2, 3])
            ## VideoName3
            video_name3 = self.annotations.iloc[index+2, 0].split("/")[-2]
            img_name3 = self.annotations.iloc[index + 2 , 0].split("/")[-1].split(".")[0]

            ## Image3
            image3 = np.array(Image.open(img_path3).convert("RGB"))
            image3 = cv2.resize(image3, (self.image_size, self.image_size))
            ## Mask3
            mask3 = np.array(Image.open(label_path3).convert("L"))
            mask3 = cv2.resize(mask3, (self.image_size, self.image_size))
            ## FlowImage3
            flow_image3 = np.array(Image.open(flow_image_path3).convert("RGB"))
            flow_image3 = cv2.resize(flow_image3, (self.image_size, self.image_size))
            ## Flo3
            flo3 = self.read_flow(flow_path3)
            flo3 = np.resize(flo3, (self.image_size, self.image_size , 2))


            #if index == 0:
            #    print(self.transform)
            if self.transform:
                
                augmentations = self.transform(image=image1,  image2=image2, image3 = image3, mask1=mask1,  mask2=mask2, mask3=image3,  flo1= flo1 , flo2 = flo2, flo3 = flo3, flow_image1 = flow_image1, flow_image2= flow_image2 , flow_image3= flow_image3 )
                image1 = augmentations["image"]
                image2 = augmentations["image2"]
                image3 = augmentations["image3"]

                mask1 = augmentations["mask1"]
                mask2 = augmentations["mask2"]
                mask3 = augmentations["mask3"]

                flo1 = augmentations["flo1"]
                flo2 = augmentations["flo2"]
                flo3 = augmentations["flo3"]

                flow_image1 = augmentations["flow_image1"]
                flow_image2 = augmentations["flow_image2"]
                flow_image3 = augmentations["flow_image3"] 
            

            
            if video_name1 != video_name2 or video_name2 != video_name3 :
                print("NOT SAME")
                return image1, mask1, flow_image1, flo1, video_name1, img_name1 , image1, mask1, flow_image1, flo1, video_name1, img_name1 , image1, mask1, flow_image1, flo1, video_name1, img_name1 
            
            elif video_name1 != video_name2 or video_name2 == video_name3 :
                return image2, mask2, flow_image2, flo2, video_name2, img_name2 , image3, mask3, flow_image3, flo3, video_name3, img_name3 , image2, mask2, flow_image2, flo2, video_name2, img_name2
            
            elif video_name1 == video_name2 or video_name2 != video_name3 :
                return image1, mask1, flow_image1, flo1, video_name1, img_name1 , image2, mask2, flow_image2, flo2, video_name2, img_name2 , image1, mask1, flow_image1, flo1, video_name1, img_name1
            
            elif video_name1 == video_name2 or video_name2 == video_name3:
                return  image1, mask1, flow_image1, flo1, video_name1, img_name1 , image2, mask2, flow_image2, flo2, video_name2, img_name2 ,  image3, mask3, flow_image3, flo3, video_name3, img_name3 


        except:
            flow_path1 = os.path.join(self.flo_dir , self.annotations.iloc[index, 0])
            img_path1 = os.path.join(self.img_dir , self.annotations.iloc[index, 1])
            label_path1 = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
            flow_image_path1 = os.path.join(self.flo_img_dir, self.annotations.iloc[index, 3])


            video_name1 = self.annotations.iloc[index, 0].split("/")[-2]
            img_name1 = self.annotations.iloc[index, 0].split("/")[-1].split(".")[0]
            ## Image
            image1 = np.array(Image.open(img_path1).convert("RGB"))
            image1 = cv2.resize(image1, (self.image_size, self.image_size))
            ## Mask
            mask1 = np.array(Image.open(label_path1).convert("L"))
            mask1 = cv2.resize(mask1, (self.image_size, self.image_size))
            ## FlowImage
            flow_image1 = np.array(Image.open(flow_image_path1).convert("RGB"))
            flow_image1 = cv2.resize(flow_image1, (self.image_size, self.image_size))
            ## Flo 
            flo1 = self.read_flow(flow_path1)
            flo1 = np.resize(flo1, (self.image_size, self.image_size , 2))

            if self.transform:
                image2 , image3 = image1 , image1
                mask2 , mask3 = mask1 , mask1
                flow_image2, flow_image3 = flow_image1, flow_image1
                flo2 , flo3 = flo1, flo1


                augmentations = self.transform(image=image1,  image2=image2, image3 = image3, mask1=mask1,  mask2=mask2, mask3=image3,  flo1= flo1 , flo2 = flo2, flo3 = flo3, flow_image1 = flow_image1, flow_image2= flow_image2 , flow_image3= flow_image3 )
                image1 = augmentations["image"]
                image2 = augmentations["image2"]
                image3 = augmentations["image3"]

                mask1 = augmentations["mask1"]
                mask2 = augmentations["mask2"]
                mask3 = augmentations["mask3"]

                flo1 = augmentations["flo1"]
                flo2 = augmentations["flo2"]
                flo3 = augmentations["flo3"]

                flow_image1 = augmentations["flow_image1"]
                flow_image2 = augmentations["flow_image2"]
                flow_image3 = augmentations["flow_image3"] 
            

            
            
            return image1, mask1, flow_image1, flo1, video_name1, img_name1 , image1, mask1, flow_image1, flo1, video_name1, img_name1 , image1, mask1, flow_image1, flo1, video_name1, img_name1 
    
        

 