#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:26:20 2017

@author: caozhenjie
"""
from __future__ import print_function, division
import os
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib

try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

#
# Getting list of image files，找到图片的位置在哪里
luna_path = "/Volumes/Python&Dragon/LUNA/"
luna_subset_path = luna_path+"subset9/"
output_path = "/Volumes/Python&Dragon/LUNA/Result/subset9/"
file_list=glob(luna_subset_path+"*.mhd")

#print("这是filelist",file_list)


#Some helper functions
"""
First we import the necessary tools and find the largest nodule in the patient scan.
We're using a pandas DataFrame named df_node to keep track of the case numbers and the node information. 
The node information is an (x,y,z) coordinate in mm using a coordinate system defined in the .mhd file.

"""

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
the mask coordinates have to match the ordering of the array coordinates
    '''
    
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

############

#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)
#
# Looping over the image files
## The locations of the nodes,找到node在哪里
df_node = pd.read_csv(luna_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna() # 所有有nodule的人的编号
#  



for fcount, img_file in enumerate(tqdm(file_list)):
    
    print("Getting mask for image file %s" % img_file.replace(luna_subset_path,""))
    
    mini_df = df_node[df_node["file"]==img_file] #所有有nodule的人的编号留下来的文件
    
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        # change to world coordinates
        #biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        itk_img = sitk.ReadImage(img_file)          # 用sitk读出来了img_file里面的图
        img_array = sitk.GetArrayFromImage(itk_img) # 再用getarray把indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():    
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]  
            # just keep 3 slices
            imgs = np.ndarray([3,height,width],dtype=np.float32)
            masks = np.ndarray([3,height,width],dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint((center-origin)/spacing)  # 每个人的图片里面 nodule center in voxel space (still x,y,z ordering)
            i = 0
            for i_z in range(int(v_center[2])-1,int(v_center[2])+2):
                mask = make_mask(center,diam,i_z*spacing[2]+origin[2],width,height,spacing,origin)
                masks[i] = mask
                imgs[i] = matrix2int16(img_array[i_z])
                i+=1
            np.save(output_path+"images_%d.npy" % (fcount) ,imgs)
            np.save(output_path+"masks_%d.npy" % (fcount) ,masks)
            
imgs = np.load(output_path+'images_0.npy')
masks = np.load(output_path+'masks_0.npy')
for i in range(len(imgs)):
    print ("这个是image的id","image %d" % i)
    print ("这是origin的坐标", origin)
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(imgs[i],cmap='gray')
    ax[0,1].imshow(masks[i],cmap='gray')
    ax[1,0].imshow(imgs[i]*masks[i],cmap='gray')
    print("这个是mask，第三张图是nodule的")
    plt.show()
    raw_input("hit enter to view the next: ")

