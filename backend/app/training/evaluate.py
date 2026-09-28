"""Evaluation CLI for the SIHPS nowcaster.

Loads a trained checkpoint (or an explicitly untrained model), rebuilds the same
chronologically-split, embargo-safe windows used for training, and reports CSI /
POD / FAR / F1, Brier score, Brier skill score, reliability statistics and -
when an MC-dropout ensemble is requested - CRPS, per hazard and per lead time.

Reporting rules
---------------
* Every report carries ``is_synthetic`` and a ``validation_status`` string.
  Metrics computed against synthetic labels describe agreement with the SIHPS
  generator, **not** forecasting skill, and the CLI says so in its output.
* Undefined metrics are reported as ``null`` with a reason and counted, never
  replaced with a flattering default.
* A checkpoint that was never trained is evaluated only when explicitly allowed
  (``--allow-untrained``) and the report states ``trained_checkpoint_loaded:
  false``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import get_settings
from app.grid import GridSpec
from app.ingestion.base import EXPERIMENTAL_DISCLAIMER, WindowSpec
from app.ingestion.synthetic import SYNTHETIC_ATTRIBUTION
from app.logging_conf import get_logger, setup_logging
from app.models.network import MultiTaskNowcastNet
from app.models.registry import load_checkpoint, resolve_backend
from app.models.uncertainty import TemperatureScaler
from app.training.dataset import build_dataloaders, build_datasets
from app.training.gate import (
    EVAL_DEMO_DIR_MARKER,
    OBSERVATIONAL,
    SYNTHETIC_EVAL,
    EvaluationBlocked,
    enforce_evaluation_gate,
)
from app.training.metrics import evaluate_fields, summarise
from app.training.train import TrainConfig, load_training_cubes, resolve_device, seed_everything
from app.training.validation_report import build_capability_report

logger = get_logger("training.evaluate")

__all__ = ["EvalConfig", "evaluate", "main"]


class EvalConfig:
    """Evaluation settings (mirrors the training data options)."""

    __slots__ = (
        "checkpoint",
        "allow_untrained",
        "backend",
        "device",
        "seed",
        "n_events",
        "seq_len",
        "horizon",
        "train_fraction",
        "val_fraction",
        "embargo_frames",
        "batch_size",
        "num_workers",
        "split",
        "thresholds",
        "mc_samples",
        "reliability_bins",
        "is_synthetic",
        "data_dir",
        "apply_calibration",
        "output",
        "mode",
    )

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            "checkpoint": None,
            "allow_untrained": False,
            "backend": "auto",
            "device": "cpu",
            "seed": 7,
            "n_events": 4,
            "seq_len": 6,
            "horizon": 6,
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "embargo_frames": None,
            "batch_size": 2,
            "num_workers": 0,
            "split": "test",
            "thresholds": {"thunderstorm": 0.5, "cloudburst": 0.5, "flood": 0.5},
            "mc_samples": 0,
            "reliability_bins": 10,
            "is_synthetic": True,
            "data_dir": None,
            "apply_calibration": True,
            "output": None,
            # Phase 8.2: observational | synthetic_demo
            "mode": SYNTHETIC_EVAL,
        }
        for key, value in {**defaults, **kwargs}.items():
            if key not in defaults:
                raise TypeError(f"unknown evaluation option {key!r}")
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


def _load_model(config: EvalConfig, device: torch.device) -> tuple[MultiTaskNowcastNet, dict[str, Any]]:
    """Load a trained checkpoint, or refuse to evaluate an untrained model.

    Refusing by default is deliberate: an untrained model produces arbitrary
    numbers, and reporting them without saying so would be misleading.
    """
    settings = get_settings()
    backend = resolve_backend(config.backend)
    if backend != "torch":
        raise RuntimeError(
            f"evaluation requires the torch backend (got {backend!r}); the NumPy reference "
            "nowcaster exposes no per-hazard logits to score"
        )
    if config.checkpoint:
        directory = Path(config.checkpoint)
        model = load_checkpoint(directory, backend="torch", map_location=str(device))
        if model is None:
            raise FileNotFoundError(f"no torch checkpoint found in '{directory}'")
        meta: dict[str, Any] = {"trained_checkpoint_loaded": True}
    else:
        if not config.allow_untrained:
            raise RuntimeError(
                "no --checkpoint given. Evaluating an untrained model produces meaningless "
                "numbers; pass --allow-untrained only to smoke-test the evaluation path."
            )
        from app.models.network import NowcastNetConfig

        model = MultiTaskNowcastNet(NowcastNetConfig.preset("lite"))
        meta = {"trained_checkpoint_loaded": False}
    meta["model_version"] = str(getattr(model, "model_version", settings.model_version))
    return model.to(device).eval(), meta


def _predict(
    model: MultiTaskNowcastNet,
    loader,
    device: torch.device,
    *,
    mc_samples: int = 0,
    scaler: TemperatureScaler | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Deterministic probabilities, targets and (optional) MC ensembles."""
    predictions: dict[str, list[np.ndarray]] = {"thunderstorm": [], "cloudburst": [], "flood": []}
    targets: dict[str, list[np.ndarray]] = {"thunderstorm": [], "cloudburst": [], "flood": []}
    ensembles: dict[str, list[np.ndarray]] = {"thunderstorm": [], "cloudburst": [], "flood": []}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            terrain = batch.get("terrain")
            terrain = None if terrain is None else terrain.to(device)
            heads = model(x, terrain)["heads"]
            fields = {
                "thunderstorm": heads.thunderstorm_prob,
                "cloudburst": heads.cloudburst_prob,
                "flood": heads.flood_prob,
            }
            for name, tensor in fields.items():
                probability = tensor.detach().cpu().numpy()
                if scaler is not None and name == "cloudburst":
                    probability = scaler.transform(probability)
                predictions[name].append(probability)
                targets[name].append(batch[name].detach().cpu().numpy())
            if mc_samples:
                # ``mc_forward`` repeats the *first* sample of its input
                # (``x[:1].expand(...)``), so it is called per sample to keep the
                # ensemble attached to the right window.
                for position in range(x.shape[0]):
                    single = x[position : position + 1]
                    single_terrain = None if terrain is None else terrain[position : position + 1]
                    ensemble = model.mc_forward(
                        single,
                        single_terrain,
                        n_samples=int(mc_samples),
                        batch_chunk=min(4, int(mc_samples)),
                    )
                    for name in predictions:
                        arr = np.asarray(ensemble[name])
                        # ``(S, 1, T, H, W)`` -> ``(1, S, T, H, W)``
                        ensembles[name].append(np.moveaxis(arr, 0, 1))
    stacked_pred = {k: np.concatenate(v) for k, v in predictions.items()}
    stacked_true = {k: np.concatenate(v) for k, v in targets.items()}
    stacked_ens = (
        {k: np.concatenate(v) for k, v in ensembles.items()}
        if mc_samples and all(ensembles.values())
        else {}
    )
    return stacked_pred, stacked_true, stacked_ens

def evaluate(config: EvalConfig) -> dict[str, Any]:
    """Run the evaluation and return the full JSON-serialisable report."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    settings = get_settings()
    grid: GridSpec = settings.grid

    model, model_meta = _load_model(config, device)
    train_config = TrainConfig(
        data_dir=config.data_dir,
        n_events=config.n_events,
        seed=config.seed,
        seq_len=config.seq_len,
        horizon=config.horizon,
        is_synthetic=config.is_synthetic,
    )
    cubes, event_ids, terrain_tensor = load_training_cubes(train_config, grid)

    # ------------------------------------------------- phase 8.2 integrity gate
    # Runs on the exact cubes before any prediction, so a blocked evaluation
    # produces no metrics and writes no report. The caller's ``is_synthetic``
    # claim is checked against the cubes' own provenance, never trusted.
    integrity = enforce_evaluation_gate(
        cubes,
        requested_mode=config.mode,
        claimed_is_synthetic=config.is_synthetic,
        output=config.output,
    )
    # The verified value replaces the caller's claim everywhere downstream.
    verified_synthetic = bool(integrity["verified_is_synthetic"])
    print(
        f"[{'SYNTHETIC DEMO' if verified_synthetic else 'OBSERVATIONAL'}] "
        f"evaluation mode={config.mode}; data class={integrity['data_class']} "
        f"(verified from provenance, not from the caller's flag)"
    )

    spec = WindowSpec(seq_len=config.seq_len, horizon=config.horizon, stride=1)
    datasets, _plans = build_datasets(
        cubes,
        spec,
        terrain=terrain_tensor,
        fractions=(
            config.train_fraction,
            config.val_fraction,
            max(0.0, 1.0 - config.train_fraction - config.val_fraction),
        ),
        embargo_frames=config.embargo_frames,
        is_synthetic=verified_synthetic,
        event_ids=event_ids,
    )
    loaders = build_dataloaders(
        datasets, batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed
    )
    if config.split not in loaders or not loaders[config.split]:
        raise ValueError(f"split {config.split!r} is empty; available splits: {sorted(loaders)}")

    scaler = _load_scaler(config)
    predictions, targets, ensembles = _predict(
        model, loaders[config.split], device, mc_samples=config.mc_samples, scaler=scaler
    )
    report = evaluate_fields(
        predictions,
        targets,
        lead_hours=list(spec.horizon_hours),
        thresholds=config.thresholds,
        n_bins=config.reliability_bins,
        ensembles=ensembles,
        # ``_predict`` stacks ensembles as ``(B, S, T, H, W)``.
        ensemble_sample_axis=1,
    )
    result: dict[str, Any] = {
        "split": config.split,
        "n_windows": len(datasets[config.split]),
        "n_samples_evaluated": int(next(iter(predictions.values())).shape[0]),
        "lead_hours": list(spec.horizon_hours),
        "thresholds": dict(config.thresholds),
        "mc_samples": int(config.mc_samples),
        "calibration_applied": scaler is not None,
        "grid": grid.to_dict(),
        "event_ids": event_ids,
        "model": model_meta,
        "is_synthetic": verified_synthetic,
        "is_synthetic_claimed_by_caller": bool(config.is_synthetic),
        "evaluation_mode": config.mode,
        "integrity": integrity,
        "skill_claim_permitted": not verified_synthetic,
        "data_source": SYNTHETIC_ATTRIBUTION if verified_synthetic else "operational observations",
        "disclaimer": EXPERIMENTAL_DISCLAIMER,
        "validation_status": (
            "SYNTHETIC DEMO - these scores measure agreement with the SIHPS synthetic "
            "generator, NOT independent forecasting skill. No observational validation "
            "has been performed. These numbers may not update official skill summaries "
            "or production model-selection criteria."
            if verified_synthetic
            else "Observational data with verified provenance. Independent validation "
            "status is set by the operator; metrics are skill estimates against real "
            "observations, not a guarantee."
        ),
        "evaluation": report,
        "summary": summarise(report),
        "config": config.to_dict(),
        # Phase 5: an explicit account of which evaluations could NOT be
        # performed, so an absent metric is never mistaken for a good one.
        "validation_capability": build_capability_report(
            real_datasets_ingested=_count_real_datasets()
        ).to_dict(),
    }
    if config.output:
        Path(config.output).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _count_real_datasets() -> int:
    """Count persisted provenance rows that are *not* synthetic.

    Used by the capability report: a synthetic dataset must never be counted as
    evidence that observational validation is possible. A database that cannot
    be reached counts as zero, which is the conservative direction.
    """
    try:
        from sqlalchemy import func, select

        from app.db.models import DatasetProvenanceRow
        from app.db.session import get_session_factory

        with get_session_factory().scope() as session:
            return int(
                session.execute(
                    select(func.count(DatasetProvenanceRow.id)).where(
                        DatasetProvenanceRow.data_class != "synthetic"
                    )
                ).scalar_one()
            )
    except Exception as exc:  # noqa: BLE001 - the report must always render
        logger.warning(
            "could not count ingested datasets", extra={"error": type(exc).__name__}
        )
        return 0


def _load_scaler(config: EvalConfig) -> TemperatureScaler | None:
    """Reuse the temperature scaler saved next to the checkpoint, if present."""
    if not config.apply_calibration or not config.checkpoint:
        return None
    from app.models.registry import load_calibration

    payload = load_calibration(Path(config.checkpoint))
    if payload and "cloudburst" in payload:
        return TemperatureScaler.from_dict(payload["cloudburst"])
    return None

def _print_report(result: dict[str, Any]) -> None:
    """Human-readable summary with the synthetic-data caveat up front."""
    print("=" * 78)
    print("SIHPS EVALUATION")
    print("=" * 78)
    if result["is_synthetic"]:
        print("[SYNTHETIC DEMO] Scores below measure agreement with the synthetic generator.")
        print("            They are NOT independent forecasting skill.")
    capability = result.get("validation_capability")
    if capability:
        print(f"[VALIDATION STATUS] {capability['status']}")
        for note in capability.get("notes", []):
            print(f"  - {note}")
        for blocked in capability.get("blocked_evaluations", []):
            print(f"  NOT EVALUATED: {blocked['evaluation']}")
            print(f"      because: {blocked['blocked_because']}")
    print(
        f"split={result['split']} windows={result['n_windows']} "
        f"samples={result['n_samples_evaluated']} mc_samples={result['mc_samples']}"
    )
    print(f"trained checkpoint: {result['model']['trained_checkpoint_loaded']}")
    print(f"{'hazard':<14}{'CSI':>9}{'POD':>9}{'FAR':>9}{'F1':>9}{'Brier':>9}{'BSS':>9}{'CRPS':>9}")
    print("-" * 78)
    for hazard, entry in result["evaluation"]["pooled"].items():
        values = [
            entry["categorical"]["csi"],
            entry["categorical"]["pod"],
            entry["categorical"]["far"],
            entry["categorical"]["f1"],
            entry["brier"]["brier"],
            entry["brier"]["brier_skill_score"],
            entry["crps"]["crps"],
        ]
        cells = [f"{v:.4f}" if isinstance(v, float) else "n/a" for v in values]
        print(f"{hazard:<14}" + "".join(f"{c:>9}" for c in cells))
    print("-" * 78)
    print("n/a = undefined; reasons are listed in the JSON report")
    print(json.dumps(result["summary"], indent=2, default=str))
    print(result["disclaimer"])


def _reject_observational(namespace, _values, _option_string=None):
    """Reject ``--observational`` outright (Phase 8.3).

    The flag used to set ``is_synthetic=False`` on whatever the loader produced,
    which is how generated data was relabelled as observational. It has no
    meaning now, and a deprecated warning would keep a harmful invocation
    *working* by silently ignoring it. Failing loudly is the safe choice; use
    ``--mode observational``, which verifies provenance from the data itself.
    """
    raise SystemExit(
        "error: --observational was removed in Phase 8.3.\n"
        "It previously only flipped a label on generated data, so it could present\n"
        "synthetic cubes as observations. Use --mode observational instead, which\n"
        "verifies the dataset's SHA-256 provenance and refuses synthetic data."
    )


class _ObservationalRemovedAction(argparse.Action):  # pragma: no cover - parser glue
    def __call__(self, parser, namespace, values, option_string=None):
        _reject_observational(namespace, values, option_string)


def build_parser() -> argparse.ArgumentParser:
    """CLI for ``sihps-evaluate`` / ``python -m app.training.evaluate``."""
    parser = argparse.ArgumentParser(
        prog="sihps-evaluate",
        description=(
            "Evaluate a SIHPS checkpoint with CSI/POD/FAR/F1, Brier, BSS, reliability and "
            "(with --mc-samples) CRPS, per hazard and per lead time. On synthetic data the "
            "scores are a pipeline demonstration, not a skill claim."
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to evaluate.")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Evaluate a freshly initialised model (smoke-test only; numbers are meaningless).",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-events", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--embargo-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--mc-samples", type=int, default=0, help="MC-dropout samples per window (enables CRPS)."
    )
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument(
        "--threshold", type=float, default=None, help="Event threshold for CSI/POD/FAR (default 0.5)."
    )
    parser.add_argument("--no-calibration", action="store_true", help="Skip the saved temperature scaler.")
    parser.add_argument("--output", default=None, help="Write the full JSON report here.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--observational",
        action=_ObservationalRemovedAction,
        nargs=0,
        help=(
            "REMOVED in Phase 8.3. It only flipped a label on generated data. "
            "Use --mode observational, which verifies provenance and refuses "
            "synthetic data."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(OBSERVATIONAL, SYNTHETIC_EVAL),
        default=SYNTHETIC_EVAL,
        help=(
            "observational: requires verified real provenance, and refuses synthetic "
            f"data. {SYNTHETIC_EVAL}: a labelled demonstration that may only write "
            f"beneath a '{EVAL_DEMO_DIR_MARKER}' directory and may not update skill "
            "summaries."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``sihps-evaluate`` console script."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    threshold = 0.5 if args.threshold is None else float(args.threshold)
    config = EvalConfig(
        checkpoint=args.checkpoint,
        allow_untrained=args.allow_untrained,
        device=args.device,
        seed=args.seed,
        n_events=args.n_events,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        embargo_frames=args.embargo_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        thresholds={"thunderstorm": threshold, "cloudburst": threshold, "flood": threshold},
        mc_samples=args.mc_samples,
        reliability_bins=args.reliability_bins,
        is_synthetic=not args.observational,
        data_dir=args.data_dir,
        apply_calibration=not args.no_calibration,
        output=args.output,
        mode=args.mode,
    )
    if config.mode == SYNTHETIC_EVAL and config.output:
        # Synthetic reports are written to a separate directory so they can never
        # overwrite, or be mistaken for, an observational evaluation.
        config.output = str(Path(config.output).parent / EVAL_DEMO_DIR_MARKER
                            / Path(config.output).name)
        print(
            f"[PHASE 8.2] synthetic-demo report will be written to {config.output}"
        )
    try:
        result = evaluate(config)
    except EvaluationBlocked as exc:
        print(exc.render())
        return 3
    _print_report(result)
    if config.output:
        print(f"\nJSON report written to {config.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
