"""Training and evaluation package for the SIHPS nowcaster.

Modules
-------
``dataset``  windowed PyTorch Dataset/DataLoader + leakage-safe temporal splits
``losses``   multi-task loss (focal BCE + weighted CE + flood BCE + KL)
``metrics``  verification metrics: CSI/POD/FAR/F1, Brier, reliability, CRPS
``train``    training CLI (``sihps-train``)
``evaluate`` evaluation CLI (``sihps-evaluate``)

Scope note
----------
Training on the synthetic generator produces a *pipeline demonstration* only.
Metrics computed on synthetic labels describe agreement with the synthetic
generator, not forecasting skill. Independent observational validation against
IMD/MOSDAC data is not available in this repository.
"""

__all__ = ["dataset", "evaluate", "losses", "metrics", "train"]
