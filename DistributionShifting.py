from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--data", type=str, default="pv")
parser.add_argument("--c", type=float, default=0.75)
parser.add_argument('--k', type=int, default=10)
opt = parser.parse_args()
E = opt.epochs
data_type = opt.data
c = opt.c
k = opt.k

gpu_id = 2


class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(self, fake='fake.pt', c=0.75, i=1, n=9864):
        val = int(n * (c**i))
        self.real = torch.load(f'real_{data_type}.pt').cuda(gpu_id)
        self.real.requires_grad = False
        self.fake = torch.load(fake).cuda(gpu_id)
        self.fake.requires_grad = False
        self.realdata = torch.cat([self.real[:val], self.fake[:-1 * val]], 0)

    def __len__(self):
        return self.realdata.shape[0]

    def __getitem__(self, item):
        return self.realdata[item]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


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


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv1d(
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
        # print(inp.shape)
        out = self.block1(inp)
        # print(out.shape)
        # out = self.block2(out)
        # print(out.shape)
        out = self.block3(out)
        # print(out.shape)
        out = self.block4(out)
        # print(out.shape)
        return torch.tanh(self.block5(out))


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        # self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv1d(256, 64, 4, 1, 0)
        self.source = nn.Linear(64 * 9, 1)

    def forward(self, inp):
        # print(inp.shape)
        out = self.block1(inp)
        # print(out.shape)
        out = self.block2(out)
        # print(out.shape)
        out = self.block3(out)
        # print(out.shape)
        # out = self.block4(out)
        out = self.block5(out)
        # print(out.shape)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.sigmoid(self.source(out))
        return source


latentdim = 24
criterionSource = nn.BCELoss()
G = Generator(in_channels=latentdim, out_channels=1).cuda(gpu_id)
D = Discriminator(in_channels=1).cuda(gpu_id)
G.apply(weights_init_normal)
D.apply(weights_init_normal)

optimizerG = optim.Adam(G.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
static_z = Variable(FloatTensor(torch.randn((36, latentdim, 1)))).cuda(gpu_id)


def sample_image(stage, epoch):
    static_sample = G(static_z).detach().cpu().squeeze()
    fig, axs = plt.subplots(6, 6, sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(static_sample)):
        axs[i].plot(range(96), static_sample[i])
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(2))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))
    fig.savefig(DIRNAME + "stage%depoch%d.png" % (stage, epoch))
    plt.close(fig)


DIRNAME = f'DistShift_{data_type}/'
os.makedirs(DIRNAME, exist_ok=True)
board = SummaryWriter(log_dir=DIRNAME)

G.load_state_dict(torch.load(f'DCGAN_{data_type}/G{E-1}.pt'))
D.load_state_dict(torch.load(f'DCGAN_{data_type}/D{E-1}.pt'))
step = 0
fake_name = f'fake_{data_type}.pt'
n = len(torch.load(f"real_{data_type}.pt"))
for i in range(1, k):
    dataloader = DataLoader(NWSDataset(fake=fake_name, c=c, i=i, n=n),
                            batch_size=256,
                            shuffle=True)
    for epoch in range(E):
        print(epoch)
        for realdata in dataloader:
            noise = 1e-5 * max(1 - ((epoch + 1) / E), 0)
            step += 1
            batch_size = realdata[0].shape[0]
            trueTensor = 0.7 + 0.5 * torch.rand(batch_size)
            falseTensor = 0.3 * torch.rand(batch_size)
            probFlip = torch.rand(batch_size) < 0.05
            probFlip = probFlip.float()
            trueTensor, falseTensor = (
                probFlip * falseTensor + (1 - probFlip) * trueTensor,
                probFlip * trueTensor + (1 - probFlip) * falseTensor,
            )
            trueTensor = trueTensor.view(-1, 1).cuda(gpu_id)
            falseTensor = falseTensor.view(-1, 1).cuda(gpu_id)
            realdata = realdata.cuda(gpu_id)
            realSource = D(realdata)
            realLoss = criterionSource(realSource,
                                       trueTensor.expand_as(realSource))
            latent = Variable(torch.randn(batch_size, latentdim,
                                          1)).cuda(gpu_id)
            fakeGen = G(latent)
            fakeGenSource = D(fakeGen.detach())
            fakeGenLoss = criterionSource(fakeGenSource,
                                          falseTensor.expand_as(fakeGenSource))
            lossD = realLoss + fakeGenLoss
            optimizerD.zero_grad()
            lossD.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 20)
            optimizerD.step()
            fakeGenSource = D(fakeGen)
            lossG = criterionSource(fakeGenSource,
                                    trueTensor.expand_as(fakeGenSource))
            optimizerG.zero_grad()
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 20)
            optimizerG.step()
            board.add_scalar('realLoss', realLoss.item(), step)
            board.add_scalar('fakeGenLoss', fakeGenLoss.item(), step)
            board.add_scalar('lossD', lossD.item(), step)
            board.add_scalar('lossG', lossG.item(), step)
        if (epoch + 1) % 50 == 0:
            torch.save(
                G.state_dict(),
                DIRNAME + "Gstage" + str(i) + 'epoch' + str(epoch) + ".pt")
            torch.save(
                D.state_dict(),
                DIRNAME + "Dstage" + str(i) + 'epoch' + str(epoch) + ".pt")
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                G.eval()
                sample_image(i, epoch)
                G.train()
    with torch.no_grad():
        G.eval()
        fsize = int((1 - (c**(i + 1))) * n / c)
        fakeSamples = G(
            Variable(torch.randn(fsize, latentdim, 1)).cuda(gpu_id))

        sums = torch.trapezoid(
            fakeSamples.squeeze(),
            dx=1 / 4).detach().cpu().numpy().argsort()[::-1].copy()
        fake_name = DIRNAME + 'fake' + str(i + 1) + '.pt'
        torch.save(fakeSamples.data[sums], fake_name)
        del fakeSamples
        G.train()