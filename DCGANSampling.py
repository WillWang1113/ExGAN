import torch
import torch.nn as nn
import time
from scipy.stats import genpareto
from torch.autograd import Variable
from torch import FloatTensor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.75)
parser.add_argument('--k', type=int, default=10)
parser.add_argument("--data", type=str, default="pv")
parser.add_argument("--taus", type=list, default=[0.2, 0.1, 0.05])
parser.add_argument("--epochs", type=int, default=300)
opt = parser.parse_args()
data_type = opt.data
taus = opt.taus
c = opt.c
k = opt.k
epochs = opt.epochs
gpu_id = 2


def convTBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm1d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels, 256, 12, 1, 0)
        # self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose1d(64, out_channels, 4, 2, 1)

    def forward(self, inp):
        out = self.block1(inp)
        # out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return torch.tanh(self.block5(out))


latentdim = 24
G = Generator(in_channels=latentdim, out_channels=1).cuda(gpu_id)

train_data = torch.load(f"real_{data_type}.pt")
code = torch.trapezoid(train_data, dx=1 / 4).squeeze() / 10
# evt_params = torch.load(f"evt_params_{data_type}.pt")
# genpareto_params = evt_params["genpareto_params"]
# threshold = evt_params["threshold"]
# rv = genpareto(*genpareto_params)

G.load_state_dict(torch.load(f'DCGAN_{data_type}/G{epochs-1}.pt'))
G.eval()

for tau in taus:
    # tau_prime = tau / (c**k)
    # val = rv.ppf(1 - tau_prime) + threshold
    num = int(len(train_data) * tau)
    val = code[num]
    print("target:\n", val)
    images = []
    count = 0
    t = time.time()
    while count < 100:
        latent = Variable(FloatTensor(torch.randn(
            (500, latentdim, 1)))).cuda(gpu_id)
        image = G(latent)
        # print(image[0])
        print((torch.trapezoid(image, dx=1 / 4) / 10).max())
        sums = (torch.trapezoid(image, dx=1 / 4) / 10) >= val

        if sums.nonzero().shape[0] > 0:
            images.append(image[sums])
            count += sums.nonzero().shape[0]
            print(count)
    print(time.time() - t)
    images = torch.cat(images, 0)[:100]
    torch.save(images, 'DCGAN' + str(tau) + f'_{data_type}.pt')
