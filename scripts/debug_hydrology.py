"""Ad-hoc hydrology debug helper (not part of the shipped test suite)."""

import numpy as np

from app.ingestion.terrain import _downstream_indices, d8_flow_direction, flow_accumulation_d8

dem = np.tile(np.linspace(100, 0, 4), (3, 1))
print("dem", dem)
d = d8_flow_direction(dem, 1000.0, 1000.0)
print("dirs", d)
dn, v = _downstream_indices(d)
print("dn", dn.reshape(d.shape))
print("valid", v.reshape(d.shape))
indeg = np.zeros(d.size, dtype=int)
np.add.at(indeg, dn[v], 1)
print("indeg", indeg.reshape(d.shape))
acc = flow_accumulation_d8(dem, d)
print("acc", acc, "sum", acc.sum(), "ncells", acc.size)

ramp = np.tile(np.linspace(1000, 0, 20), (15, 1))
micro = ramp + 1e-4 * (np.arange(15)[:, None] + 1.37 * np.arange(20)[None, :])
d2 = d8_flow_direction(micro, 2000.0, 2000.0)
a2 = flow_accumulation_d8(micro, d2)
print("ramp acc max", a2.max(), "sum", a2.sum(), "ncells", a2.size)
print("ramp last cols", a2[:, -5:])
