"""Smoke test for RiskEngine, vectorisation, and Explainability service."""

import datetime
import json
import time

import numpy as np

from app.grid import GridSpec
from app.ingestion.synthetic import build_demo_terrain
from app.models.registry import build_model
from app.services.explainability import (
    GradCAMExplainer,
    PhysicalConsistencyChecker,
    WhatIfSimulator,
)
from app.services.risk_engine import RiskEngine, RiskWeights, compound_probability


def main():
    print("==================================================")
    print("SIHPS RISK ENGINE & EXPLAINABILITY SMOKE TEST")
    print("==================================================")

    # 1. Setup Grid & Himalayan Terrain
    print("\n[1/6] Building Synthetic Himalayan Terrain...")
    t0 = time.time()
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(32, 32))
    terrain_stack, exposure = build_demo_terrain(grid, seed=42)
    print(f"  Terrain built in {time.time() - t0:.2f}s:")
    print(f"  Grid: {grid.nx}x{grid.ny}, Elevation: {terrain_stack.elevation_m.min():.1f}m - {terrain_stack.elevation_m.max():.1f}m")
    print(f"  Mean slope: {terrain_stack.slope_deg.mean():.1f} deg, Mean exposure: {exposure.mean():.3f}")

    # 2. Copula calculation verification
    print("\n[2/6] Validating Bivariate Gaussian Copula Compound Risk...")
    p_ts = np.array([0.1, 0.5, 0.8, 0.95])
    p_cb = np.array([0.2, 0.6, 0.85, 0.90])
    p_ind = p_ts * p_cb
    p_cop = compound_probability(p_ts, p_cb, rho=0.6, use_copula=True)
    print(f"  Independent joint P: {np.round(p_ind, 3)}")
    print(f"  Copula (rho=0.6) P:  {np.round(p_cop, 3)}")
    assert np.all(p_cop >= p_ind - 1e-6), "Copula joint probability should reflect positive association!"
    print("  Copula test PASSED.")

    # 3. Instantiate RiskEngine & Run Multi-Hazard Fusion
    print("\n[3/6] Running Multi-Hazard Risk Computation...")
    engine = RiskEngine(
        terrain=terrain_stack,
        exposure=exposure,
        weights=RiskWeights(thunderstorm=0.3, cloudburst=0.4, flood=0.3),
        thresholds=(0.3, 0.6, 0.85),
        copula_rho=0.6,
    )

    t_now = datetime.datetime.now(datetime.timezone.utc)
    n_leads = 4
    pred = {
        "thunderstorm": np.random.uniform(0.1, 0.8, (n_leads, 32, 32)),
        "cloudburst": np.random.uniform(0.05, 0.9, (n_leads, 32, 32)),
        "flood": np.random.uniform(0.1, 0.75, (n_leads, 32, 32)),
    }
    lead_hours = [0.5, 1.0, 1.5, 2.0]

    result = engine.compute(pred, init_time=t_now, lead_hours=lead_hours, step=1)
    summary = result.summary()
    print(f"  Risk evaluation completed:")
    print(f"  Max overall risk: {summary['max_overall_risk']:.4f}")
    print(f"  Mean overall risk: {summary['mean_overall_risk']:.4f}")
    print(f"  Top hotspot: {summary['hotspots'][0] if summary['hotspots'] else 'None'}")

    # 4. GeoJSON Polygon Vectorization
    print("\n[4/6] Vectorising Active Risk Polygons (Run-Length Scanline)...")
    t0 = time.time()
    geojson = engine.to_geojson(result, min_category=1, event_type="compound")
    features = geojson["features"]
    print(f"  Generated {len(features)} GeoJSON features in {time.time() - t0:.3f}s")
    if features:
        sample_f = features[0]
        print(f"  Sample Feature ID: {sample_f['id']}")
        print(f"  Geometry Type: {sample_f['geometry']['type']}, Coordinates: {len(sample_f['geometry']['coordinates'][0])} pts")
        print(f"  Properties: area={sample_f['properties']['area_km2']} km^2, risk_max={sample_f['properties']['risk_max']}")
    assert "type" in geojson and geojson["type"] == "FeatureCollection"
    print("  GeoJSON vectorisation PASSED.")

    # 5. Point Risk Sampling
    print("\n[5/6] Querying Point Risk at Lat=30.5, Lon=78.5...")
    pt = engine.point_risk(result, lat=30.5, lon=78.5)
    print(f"  Point Category: {pt['risk_category']}")
    print(f"  Overall Risk: {pt['overall_risk']:.4f}")
    print(f"  Hazard Breakdown: TS={pt['hazards']['thunderstorm']:.3f}, CB={pt['hazards']['cloudburst']:.3f}, Flood={pt['hazards']['flood_probability']:.3f}")
    assert pt["row"] is not None and pt["col"] is not None
    print("  Point risk query PASSED.")

    # 6. Explainability Service
    print("\n[6/6] Verifying Explainability (Grad-CAM, Saliency & What-If)...")
    model = build_model(preset="lite", backend="torch")
    explainer = GradCAMExplainer(model)
    x = np.random.uniform(0.1, 0.9, (1, 6, 12, 32, 32)).astype(np.float32)
    terrain_tensor = terrain_stack.to_feature_stack()

    attr = explainer.explain(x, terrain_tensor, hazard="cloudburst", step=0)
    print(f"  Grad-CAM Saliency generated for {attr.hazard}:")
    print(f"  Top 3 contributing channels: {attr.top_channels[:3]}")
    print(f"  Saliency map shape: {attr.gradcam_map.shape}, range: [{attr.gradcam_map.min():.2f}, {attr.gradcam_map.max():.2f}]")

    checker = PhysicalConsistencyChecker()
    x_phys = np.ones((6, 12, 32, 32)) * 250.0
    x_phys[-1, 6] = 220.0  # CTT cold
    x_phys[-1, 7] = -20.0  # cooling fast
    x_phys[-1, 9] = 45.0   # high IWV
    consistency = checker.check(pred, x_phys, terrain=terrain_stack, attributions=attr)
    print(f"  Physical Consistency Audit: consistent={consistency.consistent}, score={consistency.score:.2f}")

    simulator = WhatIfSimulator(model)
    sim_res = simulator.simulate(x, terrain_tensor, perturbations={"iwv": 0.25, "ctt": -0.15})
    print(f"  Counterfactual Simulation shifts: {sim_res['delta']}")

    print("\n==================================================")
    print("ALL SMOKE TESTS SUCCEEDED!")
    print("==================================================")


if __name__ == "__main__":
    main()
