import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from video_dataloader import customDataset, transform
from torch.utils.data import DataLoader
import time




class SIAMESE(nn.Module):
    def __init__(self):
        super(SIAMESE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1)),
            nn.Conv3d(
                in_channels=256,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            ),
            torch.nn.LeakyReLU()
        )

    def conv123456(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def forward(self, a, b):
        a = self.conv123456(a)
        b = self.conv123456(b)
        output = a + b
        return output


def pearson_correlation(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return 1 - cost


class SIAMESE_with_Hr_estimator(nn.Module):
    def __init__(self):
        super(SIAMESE_with_Hr_estimator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
            )
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1)),
            nn.Conv3d(
                in_channels=256,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            ),
            torch.nn.LeakyReLU()
        )
        self.conv7=nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=16,kernel_size=3,padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2,padding=1),
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,dilation=2,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2,padding=1),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,dilation=2,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2,padding=1),
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,dilation=2,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((30,1)),
        )

        self.fc1=nn.Sequential(
            nn.Linear(in_features=30,out_features=30),
            nn.ReLU(),
            nn.Linear(in_features=30,out_features=1),

        )


    def conv123456(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def hr_estimator(self,x):
        x=self.conv7(x)
        x=torch.squeeze(x)
        x=self.fc1(x)
        return x

    def forward(self, a, b):
        a = self.conv123456(a)
        b = self.conv123456(b)
        x = a + b
        x=torch.squeeze(x)
        c=x.shape[0]
        x=x.reshape(c,2,-1)
        output=self.hr_estimator(x)
        return output
