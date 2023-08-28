
import torch
from siamese_network import SIAMESE
import cv2
from PIL import Image
import os
from torchvision import transforms
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt



def peak_to_peak(signal):


    xx=find_peaks(signal,height=0,prominence=0.5,distance=5)

    sum=0
    for i in range(0,len(xx[0][:])-1):
        sum=sum+(xx[0][i+1]-xx[0][i])

    Hr=60/((sum/(len(xx)-1))/30)
    a=int(Hr)
    return  a

def read_image_process(img_path):
    img=Image.open(img_path)
    img = img.resize((40, 140))
    img = transform(img)
    img = img.cuda()
    img = torch.unsqueeze(img, 1)
    img = torch.unsqueeze(img, 0)

    return img


def get_rfft_hr(signal):
    minFreq = 1
    maxFreq = 3
    framerate = 5
    fft_spec = []
    signal_size = len(signal)
    signal = signal.flatten()

    fft_data = np.fft.fft(signal)  # FFT
    fft_data = np.abs(fft_data)
    freq = np.fft.fftfreq(signal_size, 1. / framerate)  # Frequency data

    inds = np.where((freq < minFreq) | (freq > maxFreq))[0]
    fft_data[inds] = 0
    bps_freq = 60.0 * freq
    max_index = np.argmax(fft_data)
    fft_data[max_index] = fft_data[max_index] ** 2
    fft_spec.append(fft_data)
    HR = bps_freq[max_index]

    print("HR:",HR)
    return HR

