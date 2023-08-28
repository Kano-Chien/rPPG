#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import torch
import argparse
import os.path
from torchvision import transforms
from siamese_network import SIAMESE
#Show prepare view
from image_process import face_detection,face_to_head_cheel_nosave
from hr_estimation import get_rfft_hr
import time
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

siamese = SIAMESE().cuda()
siamese.load_state_dict(torch.load('C:\\Users\Kano\Desktop\demo\Siamese_network -3 -0.5559573158107227.pkl'))
siamese.eval()
print(siamese)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);
count=0
hr=0
fps=0
model_time=0
while True:
    ret, frame = cap.read()
    #cv2.imshow('Frame',frame)
    start=time.time()
    face,rect_image,no_face=face_detection(frame)
    if(no_face!=True):
        head,cheek=face_to_head_cheel_nosave(face)

        head=cv2.resize(head,(140,40))
        head=transform(head)
        head=torch.unsqueeze(head,1)
        head=head.cuda()
        cheek=cv2.resize(cheek,(140,40))
        cheek=transform(cheek)
        cheek=torch.unsqueeze(cheek,1)
        cheek=cheek.cuda()


        if(count<600):
            if(count==0):
                cv2.imshow('Frame', rect_image)
                cat_head=torch.tensor(head)
                cat_cheek=torch.tensor(cheek)
                model_time=0
            else:
                cv2.imshow('Frame', rect_image)
                cat_head=torch.cat([cat_head,head],1)
                cat_cheek=torch.cat([cat_cheek,cheek],1)
                model_time=0
            count+=1
            if(count!=600):
                print(count)
        else:
            model_time_start=time.time()
            cat_head=torch.unsqueeze(cat_head,0)
            cat_cheek=torch.unsqueeze(cat_cheek,0)
            rppg=siamese(cat_head.cuda(),cat_cheek.cuda())
            rppg=torch.squeeze(rppg)

            rppg=rppg.to(device='cpu')
            rppg=rppg.detach().numpy()

            count=count-1
            hr=get_rfft_hr(rppg)
            cat_head=torch.squeeze(cat_head,0)
            cat_cheek=torch.squeeze(cat_cheek,0)
            cat_head=cat_head[:,1:600,:,:]
            cat_cheek=cat_cheek[:,1:600,:,:]

            model_time_end=time.time()

            model_time=model_time_end-model_time_start

    else:
        print("no face")

    end=time.time()
    seconds=end-start
    if(count<599):
        fps = 1 / (seconds + model_time)

        cv2.putText(rect_image, 'FPS :'+str(np.round(fps,3)), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0),2)
    elif(count==599):
        fps = 1 / (seconds + model_time)

        cv2.putText(rect_image, 'HR :' + str(np.round(hr, 3)), (10, 75), cv2.FONT_HERSHEY_TRIPLEX, 1,(0, 255, 255),2)
        cv2.putText(rect_image, 'FPS :' + str(np.round(fps, 3)), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0),2)
    else:
        #fps = 1 / (seconds + model_time)

        cv2.putText(rect_image, 'HR :' + str(np.round(hr, 3)), (10, 75), cv2.FONT_HERSHEY_TRIPLEX, 1,(0, 255, 255),2)
        cv2.putText(rect_image, 'FPS :' + str(np.round(fps, 3)), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0),2)

    cv2.imshow('Frame', rect_image)


    print(fps)

    cv2.waitKey(1)      

cv2.destroyAllWindows()
