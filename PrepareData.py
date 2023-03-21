import torch
import numpy as np
from scipy.stats import skewnorm, genpareto
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="pv")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--c", type=float, default=0.75)
opt = parser.parse_args()
data_type = opt.data
c, k = opt.c, opt.k
gpu_id = 2

X = torch.load(f'real_{data_type}.pt').cuda(gpu_id)
X_s = torch.load(f'DistShift_{data_type}/fake{k}.pt')
n = len(X)
val = int((c**k) * n)
X_concat = torch.cat([X[:val], X_s[:n - val]], 0)
# print(X_concat)

sums = torch.trapezoid(X_concat, dx=1 / 4).cpu().numpy() / 10
# print(sums)

percentile = (100 * (1 - (c**k)))
tail = np.where(sums > np.percentile(sums, percentile))[0][-1]

body_dist, tail_dist = sums[tail:], sums[:tail]
skewnorm_params = skewnorm.fit(sums)
genpareto_params = genpareto.fit(tail_dist - sums[tail])
# print(genpareto_params)
evt_params = {
    "genpareto_params": genpareto_params,
    "threshold": float(sums[tail].squeeze())
}
print(evt_params)
torch.save(evt_params, f"evt_params_{data_type}.pt")
