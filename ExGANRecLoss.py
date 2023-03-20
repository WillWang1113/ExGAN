import torch
import torch.nn as nn
from scipy.stats import genpareto
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="pv")
parser.add_argument("--epochs", type=int, default=300)
opt = parser.parse_args()
data_type = opt.data
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
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels + 1, 256, 12, 1, 0)
        # self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose1d(64, out_channels, 4, 2, 1)

    def forward(self, latent, continuous_code):
        inp = torch.cat((latent, continuous_code), 1)
        out = self.block1(inp)
        out = self.block3(out)
        out = self.block4(out)
        return torch.tanh(self.block5(out))


latentdim = 24
G = Generator(in_channels=latentdim, out_channels=1).cuda(gpu_id)

evt_params = torch.load(f"evt_params_{data_type}.pt")
genpareto_params = evt_params["genpareto_params"]
threshold = evt_params["threshold"]
rv = genpareto(*genpareto_params)

G.load_state_dict(torch.load(f'ExGAN_{data_type}/G{epochs-1}.pt'))
G.eval()
G.requires_grad = False

# TODO: TEST set
real = torch.load('test_.pt').cuda(gpu_id)
num = len(real)
z = torch.zeros((num, latentdim, 1)).cuda(gpu_id)
z.requires_grad = True

code = torch.trapezoid(real, dx=1 / 4).cpu().numpy() / 10

optimizer = torch.optim.Adam([z], lr=1e-2)
criterion = nn.MSELoss()
for i in range(2000):
    pred = G(z, code)
    loss = criterion(pred, real)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)