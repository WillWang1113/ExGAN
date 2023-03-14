import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

data = torch.load("my_real.pt")
data = data.reshape([-1, 96])
data = data.type(torch.float32)

sums = torch.trapezoid(data, dx=1 / 4, dim=-1).numpy().argsort()[::-1].copy()
print(sums)
print(data)
data = data[sums]
print(data)
# data = data.reshape([-1, 1, 96])
# torch.save(data, 'my_real.pt')
