import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

dataset= "E:\coface\cohface"
person_dir=os.listdir(dataset)
groundtruth=[]
sum_size=0
sampling_frame=256
size=[]
stack=[]
train_gt=[]
val_gt=[]
count=1
shifting=256
for i in range(1, 39):
    if (count == 24):
        count += 1
    elif (count == 26):
        count += 1
    person_path=os.path.join(dataset,str(count))
    print(person_path)
    gt_dir=os.listdir(person_path)
    for j in  range (0,len(gt_dir)):
        gt_path=os.path.join(person_path,str(j))
        frame_path=os.path.join(gt_path,'frame')
        frame_size=len(os.listdir(frame_path))

        sum_size+=frame_size
        size.append(frame_size)

        gt_path=os.path.join(gt_path,'data.hdf5')
        p1=h5py.File(gt_path,"r+")
        gt=p1['pulse']
        gt=np.asarray(gt)
        signal_index=1

        groundtruth=[]
        for k in range(0,frame_size):
            index=round(signal_index)
            groundtruth.append(gt[index])
            signal_index += 12.8

        if(i<=26):
            for k in range(0,len(groundtruth)-sampling_frame,shifting):
                stack=[]
                for l in range(k,k+sampling_frame):
                    stack.append(groundtruth[l])
                train_gt.append(stack)
            print("(train)person:{} number:{}" .format(i,j))
        else:
            for k in range(0,len(groundtruth)-sampling_frame,shifting):
                stack=[]
                for l in range(k,k+sampling_frame):
                    stack.append(groundtruth[l])
                val_gt.append(stack)
            print("(val)person{} number{}".format(i, j))
    count+=1

train_HR=np.array(train_gt)
np.save("C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_all_train_gts.npy",train_HR)

val_HR=np.array(val_gt)
np.save("C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_all_val_gts.npy",val_HR)
