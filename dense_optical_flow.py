import numpy as np
import cv2
import time

import cv2 as cv
from matplotlib.colors import hsv_to_rgb
import os

def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    assert filename.endswith('.flo')

    ## Giving "height", "width" and "data in flow" to the file
    flow = flow[ :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close() 


def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


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


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)



clean_path = r'C:\Users\azure.lavdierada\ARFlow1\Data\flow_dataset\MPI_Sintel\clean'
videos_file_path = r'C:\Users\azure.lavdierada.BAHCESEHIR\Downloads\drive-download-20230310T124748Z-001'
#videos_names = os.listdir(clean_path)
flow_image_save_dir = r"C:\Users\azure.lavdierada.BAHCESEHIR\EM-Flow-Segmentation\flow_viz"
flo_save_dir = r"C:\Users\azure.lavdierada.BAHCESEHIR\EM-Flow-Segmentation\flow"
GtMask_dir = r"C:\Users\azure.lavdierada.BAHCESEHIR\EM-Flow-Segmentation\GtMask"
image_dir  = r"C:\Users\azure.lavdierada.BAHCESEHIR\EM-Flow-Segmentation\Image"


#videos_names = ['chagas54']
videos_names  = os.listdir(videos_file_path)
videos_names = [ videos_name.split(".")[0] for videos_name in videos_names ]

for video_name in videos_names:
    video_path = videos_file_path + "\{}.MOV".format(video_name)
    #video_path = r'C:\Users\azure.lavdierada\Structural_Similiarity_UNet_KNN\videos-20221210T101512Z-001\videos\{}.MOV'.format(video_name)
    print(video_path)
    #break
    cap = cv2.VideoCapture(video_path)

    suc, prev = cap.read()

    ## Denoising the Image/Frame
    prev = cv2.resize(prev, ( 960 , 540))
    prev = cv.fastNlMeansDenoisingColored(prev,None,6,6,7,21)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    

    min_val = 0
    max_val = 0
    idx = 0
    gap = 1
    while True:
        
        suc, img = cap.read()
        if not suc or idx > 250:
            cap.release()
            #cv2.destroyAllWindows()
            print('min_val::',min_val)
            print('max_val::',max_val)
            break

        ## Denoising the Image/Frame
        img = cv2.resize(img, ( 960 , 540))
        img = cv.fastNlMeansDenoisingColored(img,None,6,6,7,21)

        ## Image to Gray Scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        # start time to calculate FPS
        start = time.time()

        
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        '''
        new_max = np.max(flow)
        new_min = np.min(flow)
        if new_min < min_val:
            min_val = new_min
        
        if new_max > max_val:
            max_val = new_max
        
        '''

        prevgray = gray


        # End time
        #end = time.time()
        # calculate the FPS for current frame detection
        #fps = 1 / (end-start)

        #print(f"{fps:.2f} FPS")

        ## Flow to image 
        hsv_flow = draw_hsv(flow)

        hsv_flow_gray = cv2.cvtColor(hsv_flow, cv2.COLOR_BGR2GRAY)

        ret4,th4 = cv.threshold(hsv_flow_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)  ## Have 0 and 255 values only
        flow_mask = th4

        #th4 = th4/255.0

        ## Save the Image
        ## Make dir for each video  
        image_directry_path = os.path.join(image_dir, video_name)
        os.makedirs(image_directry_path, exist_ok= True)
                
        ## Save the GT_Mask
        ## Make dir for each video
        gt_mask_directry_path = os.path.join(GtMask_dir, video_name)
        os.makedirs(gt_mask_directry_path, exist_ok= True)

        ## Save the flow image
        ## Make dir for each video
        flow_image_directry_path = os.path.join(flow_image_save_dir, video_name)
        os.makedirs(flow_image_directry_path, exist_ok= True)

        ## Save the .flo file ==> Making Directry
        flo_dir_path = os.path.join(flo_save_dir, video_name)
        os.makedirs(flo_dir_path, exist_ok= True)
        

        #os.makedirs(save_file_name, exist_ok= True) 
        #print(idx )

        if idx == 0:
            index = 'frame_'+ '000' +  str(idx )
            cv2.imwrite(f"{flow_image_directry_path}/{index}.png", hsv_flow)
            cv2.imwrite(f"{gt_mask_directry_path}/{index}.png", flow_mask)
            cv2.imwrite(f"{image_directry_path}/{index}.png", img)
            write_flo(flow, f"{flo_dir_path}/{index}.flo")
        else:

          if idx % gap == 0:

            if len( str (idx) )  == 1:
              
              index =  'frame_' +'000' +  str(idx )
              #print(index)
            elif len( str (idx) ) == 2:
              index =  'frame_'+ '00' +  str(idx )
              #print(index)
            elif len( str (idx) ) == 3:
              index =  'frame_'+ '0' +  str(idx )
              #print(index)

            elif len( str (idx) ) == 4:
                index =  'frame_'+  str(idx )
                #print( 'Too many number of Frames ')
            
            '''
            if video_name == 'chagas4' or 'chagas6':
                cv2.imwrite(f"{flow_image_directry_path}/{index}.jpg", hsv_flow)
                
            else:
                cv2.imwrite(f"{flow_image_directry_path}/{index}.png", hsv_flow)
            '''
            #cv2.imwrite(f"{flow_image_directry_path}/{index}.png", hsv_flow)
            #write_flo(flow, f"{flo_dir_path}/{index}.flo")

            cv2.imwrite(f"{flow_image_directry_path}/{index}.png", hsv_flow)
            cv2.imwrite(f"{gt_mask_directry_path}/{index}.png", flow_mask)
            cv2.imwrite(f"{image_directry_path}/{index}.png", img)
            write_flo(flow, f"{flo_dir_path}/{index}.flo")

        idx += 1
        



        '''
        img1 = hsv_flow
        img2 = flow_to_image(flow)

        #print(np.max(hsv_flow))
        hsv_flow = hsv_flow/np.max(hsv_flow)
        
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #print(np.unique( img, return_counts=True ))

        ## Threshold the flowq
        #img = cv.medianBlur(gray_flow,5)
        
        ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                    cv.THRESH_BINARY,11,2)
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,11,2)
        
        ret3,th3 = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)  ## Have 0 and 1 values only
        th3 = th3/255.0

        ret4,th4 = cv.threshold(img1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)  ## Have 0 and 1 values only
        th4 = th4/255.0

        #print(np.unique( th4, return_counts=True ))

        #cv2.imshow('flow', draw_flow(gray, flow))
        #cv2.imshow('THRESH_BINARY', th1)
        #cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th2)
        #cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', th3)
        
        image = np.concatenate((cv.resize(th4, (512, 512)), np.ones((512, 10)),  cv.resize(th3, (512, 512))), axis=1)
        cv2.imshow('Both_THRESH', image)
        #cv2.imshow('THRESH_OTSU', th4)
        #cv2.imshow('flow HSV', hsv_flow)
        #cv2.imshow('th3', th3)
        
        '''
        #key = cv2.waitKey(5)
        #if key == ord('q'):
        #    break

    #print(min_val)
    #print(max_val)

    #cap.release()
    #cv2.destroyAllWindows()