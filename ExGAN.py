from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import LongTensor, FloatTensor
from scipy.stats import skewnorm, genpareto
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.75)
parser.add_argument("--gpu_id", type=int, default=2)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument("--data", type=str, default="pv")
opt = parser.parse_args()
gpu_id = opt.gpu_id
data_type = opt.data
E = opt.epochs
c = opt.c
k = opt.k


class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(self, fake='DistShift/fake10.pt', c=0.75, k=5, n=9864):
        val = int((c**k) * n)
        self.real = torch.load(f'real_{data_type}.pt').cuda(gpu_id)
        self.fake = torch.load(fake).cuda(gpu_id)
        self.realdata = torch.cat([self.real[:val], self.fake[:n - val]], 0)
        indices = torch.randperm(n)
        self.labels = torch.trapezoid(self.realdata, dx=1 / 4) / 10
        self.realdata = self.realdata[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return self.realdata.shape[0]

    def __getitem__(self, item):
        img = self.realdata[item]
        lbl = self.labels[item]
        return img, lbl


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
        nn.InstanceNorm2d(out_channels),
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

    def forward(self, inp, extreme):
        sums = torch.trapezoid(inp, dx=1 / 4) / 10
        # sums = inp.sum(dim=(1, 2, 3)) / 4096
        diff = torch.abs(extreme.view(-1, 1) - sums.view(-1, 1)) / torch.abs(
            extreme.view(-1, 1))
        out = self.block1(inp)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.sigmoid(self.source(torch.cat([out, diff], 1)))
        return source


latentdim = 24
criterionSource = nn.BCELoss()
G = Generator(in_channels=latentdim, out_channels=1).cuda(gpu_id)
D = Discriminator(in_channels=1).cuda(gpu_id)
G.apply(weights_init_normal)
D.apply(weights_init_normal)

fakename = f'DistShift_{data_type}/fake{k}.pt'
data = torch.load(fakename)
real_code = torch.trapezoid(data, dx=1 / 4).cpu().numpy() / 10

# TODO: fix the fitting
evt_params = torch.load(f"evt_params_{data_type}.pt")
print(evt_params)
genpareto_params = evt_params["genpareto_params"]
threshold = evt_params["threshold"]
rv = genpareto(*genpareto_params)


def sample_genpareto(size):
    probs = torch.rand(size) * 0.95
    return FloatTensor(rv.ppf(probs)) + threshold


def sample_cont_code(batch_size):
    return Variable(sample_genpareto((batch_size, 1, 1))).cuda(gpu_id)


optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
static_code = sample_cont_code(36)


def sample_image(batches_done):
    static_z = Variable(FloatTensor(torch.randn(
        (36, latentdim, 1)))).cuda(gpu_id)
    static_sample = G(static_z, static_code).detach().cpu().squeeze()
    fig, axs = plt.subplots(6, 6, sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(len(static_sample)):
        axs[i].plot(range(96), static_sample[i])
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(2))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(2))

    fig.savefig(DIRNAME + "%d.png" % batches_done)
    plt.close(fig)


DIRNAME = f'ExGAN_{data_type}/'
os.makedirs(DIRNAME, exist_ok=True)
board = SummaryWriter(log_dir=DIRNAME)
step = 0
n = len(torch.load(f"real_{data_type}.pt"))
dataloader = DataLoader(NWSDataset(fake=fakename, c=c, k=k, n=n),
                        batch_size=256,
                        shuffle=True)
for epoch in range(0, E):
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
        images, labels = images.cuda(gpu_id), labels.view(-1, 1).cuda(gpu_id)
        realSource = D(images, labels)
        realLoss = criterionSource(realSource,
                                   trueTensor.expand_as(realSource))
        latent = Variable(torch.randn(batch_size, latentdim, 1)).cuda(gpu_id)
        code = sample_cont_code(batch_size)
        fakeGen = G(latent, code)
        fakeGenSource = D(fakeGen.detach(), code)
        fakeGenLoss = criterionSource(fakeGenSource,
                                      falseTensor.expand_as(fakeGenSource))
        lossD = realLoss + fakeGenLoss
        optimizerD.zero_grad()
        lossD.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 20)
        optimizerD.step()
        fakeGenSource = D(fakeGen, code)

        # TODO: Other different measure ?
        fakeLabels = torch.trapezoid(fakeGen, dx=1 / 4) / 10
        rpd = torch.mean(
            torch.abs(
                (fakeLabels - code.view(batch_size)) / code.view(batch_size)))
        lossG = criterionSource(fakeGenSource,
                                trueTensor.expand_as(fakeGenSource)) + rpd
        optimizerG.zero_grad()
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 20)
        optimizerG.step()
        board.add_scalar('realLoss', realLoss.item(), step)
        board.add_scalar('fakeGenLoss', fakeGenLoss.item(), step)
        board.add_scalar('fakeContLoss', rpd.item(), step)
        board.add_scalar('lossD', lossD.item(), step)
        board.add_scalar('lossG', lossG.item(), step)
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), DIRNAME + 'G' + str(epoch) + ".pt")
        torch.save(D.state_dict(), DIRNAME + 'D' + str(epoch) + ".pt")
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            G.eval()
            sample_image(epoch)
            G.train()