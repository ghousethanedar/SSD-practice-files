#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:32:03 2020

@author: ghouse thanedar

"""

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform,VOC_CLASSES as label_map
from ssd import build_ssd
import imageio


def detect(frame,net,transform):
    height,width =frame.shape[0:2]
    frame_t = transform(frame)[0]
    x=torch.from_numpy(frame_t).permute(2,0,1)
    x=Variable(x.unsqueeze(0))
    y=net(x)
    detections=y.data
    scale=torch.Tensor([width,height,width,height])
    for i in range(detections.size(1)):
        j=0
        while detections[0,i,j,0]>=0.6:
            pt=(detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2)
            cv2.putText(frame, label_map[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j+=1
    return frame


# creating the neural network

net=build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=lambda storage,loc:storage))  

# create a transform function to fit our input on the basis of the neural network
            
            
transform = BaseTransform(net.size,(104/256.0,117/256.0,123/256.0))


# reading the video and creating the output of the video

reader=imageio.get_reader('Traffic_input.mp4')
fps=reader.get_meta_data()['fps']
writer=imageio.get_writer('Traffic_output.mp4',fps=fps)

for i,frame in enumerate(reader):
    frame=detect(frame,net,transform)
    writer.append_data(frame)
    print(i)
    
writer.close()
    


          
    


