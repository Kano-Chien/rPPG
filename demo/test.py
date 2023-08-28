import torch

a=torch.ones([3,600,40,140])
b=a[:,1:600,:,:]
print(b.size())