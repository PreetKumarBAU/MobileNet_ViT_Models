import numpy as np
import os
import pathlib
import shutil
import os
import csv
#root_dir = r"C:\Users\azure.lavdierada\Downloads\dense-ulearn-vos-main\dense-ulearn-vos-main\data\chagas_train"

#root_dir = r"C:\Users\azure.lavdierada\Structural_Similiarity_UNet_KNN\ChagasVideos\val_images"
root_dir = r"C:\Users\azure.lavdierada.BAHCESEHIR\EM-Flow-Segmentation\ChagasVideoDataset"
base_name = os.path.basename(root_dir)

## Open a File in Append Mode
csv_name = "test" ## "train" "test" "val"
root_dir = r"C:\Users\azure.lavdierada\EM-Flow-Segmentation\ChagasVideoDataset" + "\{}".format(csv_name)

#csv_filename = "MySuperSplit_{}.csv".format(csv_name)
csv_filename = "{}.csv".format(csv_name)
file1 = open(csv_filename,"w", newline='')#append mode
writer = csv.writer(file1)
flag = 1

list_folders = os.listdir(root_dir)
#images_folder_name = list_folders[1]
#annotations_folder_name = list_folders[0]

images_folder_name = "Image"
annotations_folder_name = "GtMask"
flow_folder_name = "flow"
flow_viz_folder_name = "flow_viz"


#print( os.listdir( os.path.join( root_dir , annotations_folder_name ) ))

Number_Image_Folders = len( os.listdir( os.path.join( root_dir , images_folder_name ) ))
Number_Annotation_Folders = len( os.listdir( os.path.join( root_dir , annotations_folder_name ) ) )
Number_flow_Folders = len( os.listdir( os.path.join( root_dir , flow_folder_name ) ))
Number_flow_viz_Folders = len( os.listdir( os.path.join( root_dir , flow_viz_folder_name ) ) )


#Number_Annotation_Folders = 0

if Number_Annotation_Folders == 0:
    list_videos_annofolder = ['None'] * Number_Image_Folders
else:
    list_videos_annofolder = sorted( os.listdir( os.path.join( root_dir , annotations_folder_name ) ) )

list_videos_imagefolder = sorted(os.listdir( os.path.join( root_dir , images_folder_name ) ) )
list_videos_flofolder = sorted(os.listdir( os.path.join( root_dir , flow_folder_name ) ) )
list_videos_flow_viz_folder = sorted(os.listdir( os.path.join( root_dir , flow_viz_folder_name ) ) )

print("list_videos_imagefolder:::",list_videos_imagefolder)
#row = ['Flow','Image','GtMask', 'FlowImage']
#writer.writerow(row)
import random
for (video_name_Jpeg , video_name_anno,  video_name_flo , video_name_flow_viz) in zip(list_videos_imagefolder, list_videos_annofolder , list_videos_flofolder, list_videos_flow_viz_folder ):
    
    image_filenames = sorted ( os.listdir( os.path.join( root_dir , images_folder_name , video_name_Jpeg )) )
    flo_filenames = sorted ( os.listdir( os.path.join( root_dir , flow_folder_name , video_name_flo )) )
    flow_viz_filenames = sorted ( os.listdir( os.path.join( root_dir , flow_viz_folder_name , video_name_flow_viz )) )
    #mask_filenames = sorted ( os.listdir( os.path.join( root_dir , annotations_folder_name , video_name_anno )) )


    num_image_frames = len(image_filenames)
    if Number_Annotation_Folders == 0:
        anno_filenames = ['None'] * num_image_frames
    else:
        anno_filenames = sorted ( os.listdir( os.path.join( root_dir , annotations_folder_name , video_name_anno )) )

    print('video name',video_name_Jpeg)
    print( "Number of image_filenames:::", len( image_filenames) )
    print( "Number of anno_filenames:::", len(anno_filenames) )
    print( "Number of flo_filenames:::", len(flo_filenames) )
    print( "Number of flow_viz_filenames:::", len(flow_viz_filenames) )

    frame_number = []
    '''
    rand_int = random.randint(0,len( image_filenames) ) 
    while   rand_int + 201 > len( image_filenames):
        rand_int = random.randint(0,len( image_filenames) )

    rand_int_range = list(np.arange(rand_int, rand_int + 100 ))
    '''



    for i, (image_filename, anno_filename, flo_filename , flow_viz_filename) in enumerate( zip(image_filenames , anno_filenames, flo_filenames, flow_viz_filenames )):

    #if i in rand_int_range:

        #frame_number.append( int( image_filename.split(".")[0].split("frame_")[1] ) )
        
        #anno_filename = anno_filename.split(".")[0] + '.txt'
        image_path = os.path.join( base_name, csv_name, images_folder_name, video_name_Jpeg ,  image_filename)
        anno_path = os.path.join( base_name, csv_name, annotations_folder_name, video_name_anno ,  anno_filename)
        
        flo_path = os.path.join( base_name, csv_name, flow_folder_name, video_name_flo ,  flo_filename)
        flow_viz_path = os.path.join( base_name, csv_name, flow_viz_folder_name, video_name_flow_viz ,  flow_viz_filename)

        image_path = image_path.replace(  '\\' , '/' )
        anno_path = anno_path.replace(  '\\' , '/' )
        flo_path = flo_path.replace(  '\\' , '/' )
        flow_viz_path = flow_viz_path.replace(  '\\' , '/' )

        #print(image_path)
        #print(anno_path)
        
        #print("image_path::" ,image_path )
        #image_path_ = "./data/" + image_path
    
    #start_frame_number = min(frame_number)
    #end_frame_number   = max(frame_number)
        #row = [image_path , anno_path, flo_path , flow_viz_path]
        #row = [image_path , anno_path, flo_path ]
        row = [flo_path , image_path, anno_path , flow_viz_path ]
        writer.writerow(row)
    #file1.write("{} {}\n".format( image_path , anno_path  ))     
    '''
        if Number_Annotation_Folders == 0:
            file1.write("{} \n".format( image_path  ))
            

        else:
            
            anno_path = os.path.join( base_name, annotations_folder_name, video_name_anno ,  anno_filename )
            anno_path = anno_path.replace(  '\\' , '/' ) 
            #print(anno_path)
            if 'png' in anno_path:
                
                anno_path = anno_path.replace(  'png' , 'jpg' ) 
                #print("True")
            
            #anno_path_ = "./data/" + anno_path

        #print(image_path_)
        #print("EXECUTED")
            file1.write("{} {} {}\n".format(flag, image_path_ ,anno_path_ ))
            #print(image_path_)
            #file1.write("{} {}\n".format( image_path ,anno_path ))
    ''' 

        
        
        #if i == 9:
        #    break
    

        #print("path:" , path)
        
    

file1.close()


### Read File after Appending

file1 = open(csv_filename,"r")

print("Output of Readlines after appending") 
#print(file1.readlines())
file1.close()


