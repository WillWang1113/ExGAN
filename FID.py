import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy import linalg
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="pv")
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--model", type=str, default="ExGAN")
opt = parser.parse_args()
data_type = opt.data
tau = opt.tau
model = opt.model
# if data_type == 'wind':
#     if tau == 0.2:
#         val = 0.9230  # tau = 0.2
#     elif tau == 0.1:
#         val = 1.2507  # tau = 0.1
#     elif tau == 0.05:
#         val = 1.5123  # tau = 0.05
# else:
#     if tau == 0.2:
#         val = 0.4981  # tau = 0.2
#     elif tau == 0.1:
#         val = 0.5409  # tau = 0.1
#     elif tau == 0.05:
#         val = 0.5734  # tau = 0.05

# TODO: Test set
data = torch.load(f'test_{data_type}.pt')
sums = torch.trapezoid(data, dx=1 / 4) / 10
# sums = sums >= val
# data = data[sums]
num = int(tau * len(data))
data = data[:num]

G_data = torch.load(f"logs/{model}_{data_type}_np.pt")
# G_data = torch.load(f"{model}{tau}_{data_type}.pt")
G_data = torch.from_numpy(G_data).cuda(2)
numSamples = len(data)
EPOCHS = 500
loss_func = nn.L1Loss()


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(96, 128), nn.ReLU(True),
                                     nn.Dropout())
        self.decoder = nn.Sequential(nn.Linear(128, 96))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


ae = AutoEncoder().cuda()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
data = data.reshape(data.shape[0], -1)[:numSamples]
losses = []

for epoch in range(EPOCHS):
    x = torch.autograd.Variable(data[torch.randperm(numSamples)]).cuda()
    optimizer.zero_grad()
    pred = ae(x)
    loss = loss_func(pred, x)
    losses.append(loss.cpu().data.item())
    loss.backward()
    optimizer.step()
# plt.plot(losses)

ae.eval()


def FID(mu1, mu2, sigma1, sigma2):
    eps = 1e-30
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean


def calcFID(data):
    data = data.reshape(data.shape[0], -1)
    features = ae.encoder(data.cuda()).detach().cpu().numpy()
    mean, covar = np.mean(features, 0), np.cov(features, rowvar=False)
    return FID(mean, base_mean, covar, base_covar)


base_features = ae.encoder(Variable(data).cuda()).detach().cpu().numpy()
base_mean, base_covar = np.mean(base_features, 0), np.cov(base_features,
                                                          rowvar=False)

print(calcFID(G_data))