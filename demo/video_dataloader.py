import torch
from torch.utils.data import DataLoader
import imageio
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.utils.data.dataset

transform = transforms.Compose([transforms.ToTensor()])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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












