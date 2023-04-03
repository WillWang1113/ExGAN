# EGAN

Implementation of

- Extreme Scenario Generation for Renewable Energies. 

This work focuses on how to generate the extreme scenarios with the pre-defined extremeness metrics (in this paper, daily electricity generation).

This work is highly inspired by the [ExGAN: Adversarial Generation of Extreme Samples](https://arxiv.org/pdf/2009.08454.pdf). 
We add the benchmark of conditional DCGAN and post analysis in the context of renewable energies.


## Getting Started

### Reproducing the Experiments

The whole pipline is integrated into:

```
python run.py
```
If ones want to seperately run the parts of the experiments, just python them. The args are writen in the top of the scripts.

### Evaluation and Visualizing the Results

To generate samples with the trained unconditional generators:
```
python DCGANSampling.py
```

To generate samples with the trained conditional generators:
```
python CGANSampling.py
```

To compute the metrics (Reconstruction Loss and FID), turn to the corresponding scripts.

We also provide two notebooks to do the EVT analysis and post analysis.