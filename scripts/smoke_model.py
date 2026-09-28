"""Smoke test for the model architecture (timings and tensor shapes)."""

import time

import torch

from app.models.backbone import BackboneConfig, CNNConvLSTMBackbone
from app.models.network import MultiTaskNowcastNet, NowcastNetConfig
from app.models.transformer import SpatioTemporalTransformer, TransformerConfig

x = torch.rand(2, 6, 12, 32, 40)
terrain = torch.rand(2, 4, 32, 40)

backbone = CNNConvLSTMBackbone(BackboneConfig.preset("lite"))
out = backbone(x)
print("backbone features", tuple(out.features.shape), "kl", None if out.kl is None else float(out.kl))

model = MultiTaskNowcastNet(NowcastNetConfig.preset("lite"))
print(model.summary())
t0 = time.time()
res = model(x, terrain)
print("forward", round(time.time() - t0, 2), "s")
heads = res["heads"]
print("ts", tuple(heads.thunderstorm_prob.shape), "rain", tuple(heads.rain_prob.shape))
print("cb", tuple(heads.cloudburst_prob.shape), "flood", tuple(heads.flood_prob.shape))
print("attn", tuple(heads.attention_thunderstorm_to_cloudburst.shape))

t0 = time.time()
det = model.deterministic_forward(x[:1], terrain[:1])
print("deterministic", round(time.time() - t0, 2), "s", {k: v.shape for k, v in det.items() if hasattr(v, "shape")})

t0 = time.time()
mc = model.mc_forward(x[:1], terrain[:1], n_samples=6, batch_chunk=3)
print("mc(6)", round(time.time() - t0, 2), "s", mc["thunderstorm"].shape)
print("mc std>0:", bool(mc["thunderstorm"].std(axis=0).max() > 0))

vit = SpatioTemporalTransformer(TransformerConfig(embed_dim=32, depth=2, num_heads=4, patch_size=4, out_channels=8))
feats = vit(x)
print("transformer", tuple(feats.shape))
