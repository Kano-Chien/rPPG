import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.utils.data.dataset
import os
import random
from image_process import face_detection,face_to_head_cheel_nosave

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class customDataset_training(Dataset):
    def __init__(self, root_forehead, root_check, root_gts, split, transform):
        #
        # initial paths, transform,
        #

        self.transform = transform
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load img and gt path
        self.foreheads = np.load(root_forehead)
        self.check = np.load(root_check)
        self.gts = np.load(root_gts)


        print('Total data in {} split: {}'.format(split, len(self.foreheads)))

    def __getitem__(self, index):
        # read file preprocess np -> tensor return forehead and check  30->1
        # has got index
        # first read first tensor in image 1
  # from paths get every 30 path
        window=256
        count=0
        file_path=self.foreheads[index]
        frame=os.path.join(file_path,'frame')
        #forehead_path=os.path.join(file_path,'head')
        #cheek_path=os.path.join(file_path,'cheek')
        frame_path=os.path.join(file_path,'frame')
        frame_len=len(os.listdir(frame))
        rand=random.randint(0,frame_len-window)
        frame_img_path = os.path.join(frame_path, str(rand) + '.jpg')
        frame_img=cv2.imread(frame_img_path)
        x,y,w,h,noface=face_detection(frame_img)
        while(noface==True):
            rand = random.randint(0, frame_len - window)
            frame_img_path = os.path.join(frame_path, str(rand) + '.jpg')
            frame_img = cv2.imread(frame_img_path)
            x, y, w, h, noface = face_detection(frame_img)

        gt_path=os.path.join(file_path,'data.hdf5')
        p1=h5py.File(gt_path,"r+")
        gt=p1['pulse']
        gt=np.asarray(gt)
        signal_index=1

        groundtruth=[]
        for k in range(0,frame_len):
            index=round(signal_index)
            groundtruth.append(gt[index])
            signal_index += 12.8


        for i in range(rand,rand+window):
            #forehead_img_path=os.path.join(forehead_path,str(i)+'.jpg')
            #cheek_img_path=os.path.join(cheek_path,str(i)+'.jpg')
            frame_img_path=os.path.join(frame_path,str(i)+'.jpg')
            frame_img=Image.open(frame_img_path)
            face=frame_img.crop((x,y,x+w,y+h))
            forehead_img,cheek_img=face_to_head_cheel_nosave(face)

            #forehead_img=Image.open(forehead_img_path)
            forehead_img = forehead_img.resize((40, 140))
            #cheek_img=Image.open(cheek_img_path)
            cheek_img = cheek_img.resize((40, 140))

            if self.transform is not None:
                forehead_img = self.transform(forehead_img)
                forehead_img = torch.unsqueeze(forehead_img, 3)
                cheek_img = self.transform(cheek_img)
                cheek_img = torch.unsqueeze(cheek_img, 3)

            if count == 0:
                forehead = forehead_img
                cheek = cheek_img
            else:
                forehead = torch.cat([forehead, forehead_img], 3)
                cheek = torch.cat([cheek, cheek_img], 3)
            count+=1
        gt=groundtruth[rand:rand+window]
        gt=np.asarray(gt)
        gt = torch.tensor(gt)
        forehead=forehead.permute(0,3,2,1)
        cheek=cheek.permute(0,3,2,1)

        return forehead,cheek,gt

    def __len__(self):
        # return len of imgs
        return len(self.foreheads)



class customDataset(Dataset):
    def __init__(self, root_forehead, root_check, root_gts, split, transform):
        #
        # initial paths, transform,
        #

        self.transform = transform
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load img and gt path
        self.foreheads = np.load(root_forehead)
        self.check = np.load(root_check)
        self.gts = np.load(root_gts)


        print('Total data in {} split: {}'.format(split, len(self.foreheads)))

    def __getitem__(self, index):
        # read file preprocess np -> tensor return forehead and check  30->1
        # has got index
        # first read first tensor in image 1
  # from paths get every 30 path

        i=0
        for i, path in enumerate(self.foreheads[index]):  # from paths get every 30 path
            forehead_img = Image.open(self.foreheads[index][i])
            forehead_img = forehead_img.resize((40, 140))
            check_img = Image.open(self.check[index][i])
            check_img = check_img.resize((40, 140))
            if self.transform is not None:
                forehead_img = self.transform(forehead_img)
                forehead_img = torch.unsqueeze(forehead_img, 3)
                check_img = self.transform(check_img)
                check_img = torch.unsqueeze(check_img, 3)
            if i == 0:
                forehead = forehead_img
                check = check_img
            else:
                forehead = torch.cat([forehead, forehead_img], 3)
                check = torch.cat([check, check_img], 3)

        gt = self.gts[index]
        forehead=forehead.permute(0,3,2,1)
        check=check.permute(0,3,2,1)
        return forehead, check, gt

    def __len__(self):
        # return len of imgs
        return len(self.foreheads)






















