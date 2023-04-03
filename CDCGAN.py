from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import LongTensor, FloatTensor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--data", type=str, default="pv")
opt = parser.parse_args()
E = opt.epochs
data_type = opt.data
gpu_id = 2


class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(self, path='./', dsize=1644):
        self.real = torch.load(path + f'real_{data_type}.pt').cuda(gpu_id)
        # self.indices = np.random.permutation(dsize)
        self.real.requires_grad = False
        self.labels = torch.trapezoid(self.real, dx=1 / 4) / 10
        self.labels = self.labels.unsqueeze(-1)

    def __len__(self):
        return self.real.shape[0]

    def __getitem__(self, item):
        return self.real[item], self.labels[item]
        # return self.real[self.indices[item]]


dataloader = DataLoader(NWSDataset(), batch_size=256, shuffle=True)


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
        self.block1 = convTBNReLU(in_channels + 1, 256, 12, 1, 0)
        # self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose1d(64, out_channels, 4, 2, 1)

    def forward(self, inp, code):
        out = torch.concat([inp, code], dim=1)
        out = self.block1(out)
        # out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
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
        self.source = nn.Linear(64 * 9 + 1, 1)

    def forward(self, inp, exp):
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
        out = torch.concat([out, exp.view(size, -1)], dim=1)
        
        source = torch.sigmoid(self.source(out))
        return source


latentdim = 24
criterionSource = nn.BCELoss()
G = Generator(in_channels=latentdim, out_channels=1).cuda(gpu_id)
D = Discriminator(in_channels=1).cuda(gpu_id)
G.apply(weights_init_normal)
D.apply(weights_init_normal)


def sample_cont_code(batch_size):
    if data_type == "pv":
        coef = 0.8
    else:
        coef = 2.5
    return Variable(torch.rand((batch_size, 1, 1)) * coef).cuda(gpu_id)


optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
static_z = Variable(FloatTensor(torch.randn((36, latentdim, 1)))).cuda(gpu_id)
static_code = sample_cont_code(36)


def sample_image(batches_done):
    static_sample = G(static_z, static_code).detach().cpu().squeeze()
    fig, axs = plt.subplots(6, 6, sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(static_sample)):
        axs[i].plot(range(96), static_sample[i])
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(2))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))

    fig.savefig(DIRNAME + "%d.png" % batches_done)
    plt.close(fig)


DIRNAME = f'CDCGAN_{data_type}/'
os.makedirs(DIRNAME, exist_ok=True)

board = SummaryWriter(log_dir=DIRNAME)

step = 0
for epoch in range(E):
    print(epoch)
    for images, labels in dataloader:
        noise = 1e-5 * max(1 - ((epoch + 1) / E), 0)
        step += 1
        batch_size = images.shape[0]
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
        images = images.cuda(gpu_id)
        realSource = D(images + noise * torch.randn_like(images).cuda(gpu_id),
                       labels)
        realLoss = criterionSource(realSource,
                                   trueTensor.expand_as(realSource))
        latent = Variable(torch.randn(batch_size, latentdim, 1)).cuda(gpu_id)
        code = sample_cont_code(batch_size)
        fakeData = G(latent, code)
        # print(fakeData.shape)
        fakeSource = D(fakeData.detach(), code)
        fakeLoss = criterionSource(fakeSource,
                                   falseTensor.expand_as(fakeSource))
        lossD = realLoss + fakeLoss
        optimizerD.zero_grad()
        lossD.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 20)
        optimizerD.step()
        fakeSource = D(fakeData, code)

        trueTensor = 0.9 * torch.ones(batch_size).view(-1, 1).cuda(gpu_id)
        lossG = criterionSource(fakeSource, trueTensor.expand_as(fakeSource))
        optimizerG.zero_grad()
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 20)
        optimizerG.step()
        board.add_scalar('realLoss', realLoss.item(), step)
        board.add_scalar('fakeLoss', fakeLoss.item(), step)
        board.add_scalar('lossD', lossD.item(), step)
        board.add_scalar('lossG', lossG.item(), step)
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), DIRNAME + "G" + str(epoch) + ".pt")
        torch.save(D.state_dict(), DIRNAME + "D" + str(epoch) + ".pt")
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            G.eval()
            sample_image(epoch)
            G.train()
