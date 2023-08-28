#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import cv2
import numpy as np
import json
import math
import torch
import sys
import torchvision.transforms as transforms
import PIL.Image
import argparse
import os.path
from image_process import face_detection,face_to_head_cheel_nosave
#Show prepare view
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

while True:

    ret, frame = cap.read()
    start = time.time()
    face,rect_image,no_face=face_detection(frame)
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)      
    final = time.time()
    fps=1/(final-start)
    print(fps)
cv2.destroyAllWindows()
