import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from video_dataloader import customDataset, transform,customDataset_training
from torch.utils.data import DataLoader
import time
#from siamese_network import SIAMESE, pearson_correlation
from siamese_network_no_share import SIAMESE_no_share
from siamese_network import SIAMESE,SIAMESE_with_Hr_estimator
import pandas as pd
import cmath
from torch.utils.tensorboard import SummaryWriter

class pearson_correlation(nn.Module):
    def __init__(self):
        super(pearson_correlation, self).__init__()

    def forward(self,x,y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1-cost


EPOCH = 200
BATCH_SIZE = 3
LR = 0.0001
# give forehead check and PPG
loss=pearson_correlation()
if __name__ == '__main__':


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    train_loader = customDataset_training(
        root_forehead='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_train.npy', root_check='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_train_cheek.npy',
        root_gts='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_all_train_gts.npy', split='train'
        , transform=transform)
    val_loader = customDataset_training(
        root_forehead='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_val.npy', root_check='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_val_cheek.npy',
        root_gts='C:\\Users\Kano\Desktop\RGB\\111\save_npy/cohface_all_val_gts.npy', split='valid'
        , transform=transform)

    train_dataloader = DataLoader(train_loader, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
    val_dataloader = DataLoader(val_loader, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)

    #siamese = SIAMESE_no_share().cuda()
    siamese = SIAMESE().cuda()
    #siamese.load_state_dict(torch.load('C:\\Users\Kano\Desktop\RGB\\111\model\\Siamese_noShareWeights -108 -0.5498320460319519.pkl'))
    siamese.train()

    print(siamese)


    writer=SummaryWriter(log_dir="C:\\Users\Kano\Desktop\RGB\\111\\runs\\20210401(random) WINDOW=600")
    optimizer = torch.optim.Adam(siamese.parameters(), lr=LR)
    loss_count=[]

    for epoch in range(1,EPOCH):
        step = 0
        for l in range(0,10):
            for forehead, check, gt in train_dataloader:
                step = step + 1
                forehead=forehead.cuda()
                check=check.cuda()
                output = siamese(forehead, check)
                output = output.squeeze()
                train_loss = loss(output, gt.to(device=device,dtype=torch.float))
                #train_loss = pearson_correlation(output, gt.cuda().float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                writer.add_scalar('training loss',train_loss,epoch)
                print('Epoch:{} | num: {} / {} | Negative pearson correlation loss: {} '.format(epoch, step, len(train_dataloader)*10, train_loss))

        val_step = 0
        sum_val_loss = 0
        siamese.eval()
        for l in range(0,20):
            with torch.no_grad():
                for val_forehead, val_check, gt in val_dataloader:
                    val_step = val_step + 1
                    val_check=val_check.cuda()
                    val_forehead=val_forehead.cuda()
                    test_output = siamese(val_forehead, val_check)
                    test_output = test_output.squeeze()

                    val_loss = loss(test_output, gt.cuda().float())
                    #val_loss = pearson_correlation(test_output, gt.cuda().float())
                    sum_val_loss = sum_val_loss + val_loss
                    print('(val)Epoch:{} | num:{}/{} | validation_loss:{}'.format(epoch,val_step,len(val_dataloader)*20,val_loss))

        mean_val_loss = sum_val_loss/val_step
        siamese.train()
        torch.save(siamese.state_dict(), './model/Siamese_noShareWeights -{} -{}.pkl'.format(epoch, mean_val_loss.item()))
        loss_count.append(float(mean_val_loss))
        writer.add_scalar('val loss',mean_val_loss,epoch)

        test = pd.DataFrame(columns=['loss'], data=loss_count)
        test.to_csv("loss_count.csv")


