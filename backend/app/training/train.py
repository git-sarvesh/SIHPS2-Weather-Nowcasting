"""Training loop and CLI for the SIHPS multi-hazard nowcaster.

The loop reuses the existing architecture unchanged: the CNN-ConvLSTM backbone,
the multi-task heads with Cross-Hazard Attention, the variational bottleneck and
the terrain-aware flood head all come from :mod:`app.models`. Training only
supplies data, a loss, an optimiser and bookkeeping.

Optimisation uses AdamW with gradient clipping and, when a validation split
exists, keeps the checkpoint with the best validation loss.

Honesty rules enforced here
----------------------------
* Every artefact records ``is_synthetic`` and the dataset validation status.
* A run on synthetic data is labelled as a *pipeline demonstration*; the CLI
  prints and stores that it is **not** an operational forecast and that no
  accuracy has been validated.
* A checkpoint is only marked ``trained`` when at least one optimisation step
  completed, so the API cannot mistake a fresh model for a trained one.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.config import get_settings
from app.grid import FRAME_MINUTES, GridSpec
from app.ingestion.base import EXPERIMENTAL_DISCLAIMER, ObservationCube, WindowSpec
from app.ingestion.synthetic import (
    SYNTHETIC_ATTRIBUTION,
    build_demo_terrain,
    default_event_catalog,
    list_demo_events,
    load_demo_cube,
)
from app.logging_conf import get_logger, setup_logging
from app.models.network import MultiTaskNowcastNet, NowcastNetConfig
from app.models.registry import save_checkpoint
from app.models.uncertainty import TemperatureScaler
from app.training.dataset import (
    build_dataloaders,
    build_datasets,
    split_summary,
)
from app.training.gate import (
    DEMO_DIR_MARKER,
    PRODUCTION,
    SYNTHETIC_DEMO,
    TrainingBlocked,
    enforce_training_gate,
    synthetic_demo_output_dir,
)
from app.training.losses import LossWeights, MultiTaskLoss, class_balanced_weights
from app.training.metrics import evaluate_fields, summarise

logger = get_logger("training.train")

__all__ = ["TrainConfig", "TrainResult", "seed_everything", "run_epoch", "train", "main"]


@dataclass(slots=True)
class TrainConfig:
    """Everything a training run needs; all fields are CLI-overridable."""

    # data
    data_dir: str | None = None            # generated demo dataset directory
    n_events: int = 4
    seed: int = 7
    seq_len: int = 6
    horizon: int = 6
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    embargo_frames: int | None = None
    batch_size: int = 2
    num_workers: int = 0
    # model / optimisation
    preset: str = "lite"
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    pos_weight: float = 1.0
    kl_weight: float = 1e-3
    use_cha: bool = True
    # runtime
    device: str = "cpu"
    max_steps: int | None = None           # cap per epoch (CPU smoke runs)
    output_dir: str = "data/models/sihps-convlstm-cha-v0.1.0"
    log_every: int = 1
    is_synthetic: bool = True
    fit_rain_class_weights: bool = True
    # ------------------------------------------------------------- phase 8.1
    #: ``production`` or ``synthetic_demo``. There is no default that lets
    #: generated data reach a production checkpoint, and no flag that does.
    mode: str = SYNTHETIC_DEMO
    #: CAPE/IWV validation status, or ``None`` to run the validators here.
    #: ``"run"`` means "compute it now"; a real status string is passed through.
    cape_validation: str | None = "run"
    iwv_validation: str | None = "run"


@dataclass(slots=True)
class TrainResult:
    """Outcome of a training run (also written to ``training_report.json``)."""

    output_dir: str
    epochs: int
    n_parameters: int
    best_val_loss: float | None
    history: list[dict[str, Any]] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] | None = None
    calibration: dict[str, float] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run is reproducible."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - CPU-only test environment
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    """Honour an explicit request but fall back to CPU when unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "mps" and not getattr(torch.backends, "mps", None):  # pragma: no cover
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def load_training_cubes(config: TrainConfig, grid: GridSpec) -> tuple[list[ObservationCube], list[str], np.ndarray]:
    """Load (or generate) the training cubes plus the shared terrain tensor.

    A pre-generated dataset under ``data_dir`` is reused when present so a
    training run is reproducible without regenerating the terrain; otherwise the
    synthetic generator produces the sequence. Either way the cubes and the
    report state that the data is synthetic.
    """
    terrain, exposure = build_demo_terrain(grid, seed=config.seed)
    terrain_tensor = terrain.to_feature_stack()

    if config.data_dir:
        events = list_demo_events(config.data_dir)
        if events:
            cubes: list[ObservationCube] = []
            event_ids: list[str] = []
            for entry in events[: max(1, int(config.n_events))]:
                cube = load_demo_cube(config.data_dir, str(entry["event_id"]))
                if cube.grid.to_dict() != grid.to_dict():
                    raise ValueError(
                        f"cube {entry['event_id']} grid {cube.grid.to_dict()} != configured {grid.to_dict()}"
                    )
                cubes.append(cube)
                event_ids.append(str(entry["event_id"]))
            logger.info("loaded demo cubes", extra={"n": len(cubes), "data_dir": config.data_dir})
            return cubes, event_ids, terrain_tensor

    from app.ingestion.synthetic import SyntheticINSATConnector

    connector = SyntheticINSATConnector(grid, terrain)
    cubes = []
    event_ids = []
    for spec in default_event_catalog()[: max(1, int(config.n_events))]:
        cubes.append(connector.fetch(spec, exposure=exposure))
        event_ids.append(spec.event_id)
    logger.info("generated synthetic training cubes", extra={"n": len(cubes)})
    return cubes, event_ids, terrain_tensor


def run_epoch(
    model: MultiTaskNowcastNet,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    *,
    optimiser: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """One pass over ``loader``.

    With an ``optimiser`` this trains (forward, loss, backward, clip, step);
    without one it only evaluates under ``torch.no_grad()``.
    """
    training = optimiser is not None
    model.train(training)
    totals = {"loss": 0.0, "n_batches": 0}
    term_totals: dict[str, float] = {}
    started = time.time()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = batch["x"].to(device)
        terrain = batch.get("terrain")
        terrain = None if terrain is None else terrain.to(device)
        targets = {name: batch[name].to(device) for name in ("thunderstorm", "rain_class", "cloudburst", "flood", "flood_soft")}
        with torch.set_grad_enabled(training):
            outputs = model(x, terrain)
            loss, terms = criterion(outputs, targets)
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimiser.step()
        else:
            grad_norm = torch.zeros(())
        totals["loss"] += float(loss.detach())
        totals["n_batches"] += 1
        for name, value in terms.items():
            term_totals[name] = term_totals.get(name, 0.0) + value
        if not np.isfinite(float(loss.detach())):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach())}")
    n = max(1, totals["n_batches"])
    result = {
        "loss": totals["loss"] / n,
        "n_batches": totals["n_batches"],
        "seconds": round(time.time() - started, 3),
        "grad_norm": float(grad_norm),
        "terms": {name: value / n for name, value in term_totals.items()},
    }
    model.train(False)
    return result


@torch.no_grad()
def collect_predictions(
    model: MultiTaskNowcastNet, loader: DataLoader, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run the model over ``loader`` and return stacked probabilities/targets."""
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {"thunderstorm": [], "cloudburst": [], "flood": []}
    targets: dict[str, list[np.ndarray]] = {"thunderstorm": [], "cloudburst": [], "flood": []}
    for batch in loader:
        x = batch["x"].to(device)
        terrain = batch.get("terrain")
        terrain = None if terrain is None else terrain.to(device)
        outputs = model(x, terrain)
        heads = outputs["heads"]
        for name, tensor in (
            ("thunderstorm", heads.thunderstorm_prob),
            ("cloudburst", heads.cloudburst_prob),
            ("flood", heads.flood_prob),
        ):
            predictions[name].append(tensor.detach().cpu().numpy())
            targets[name].append(batch[name].detach().cpu().numpy())
    stacked_pred = {k: np.concatenate(v) for k, v in predictions.items()}
    stacked_true = {k: np.concatenate(v) for k, v in targets.items()}
    return stacked_pred, stacked_true


def _build_model_and_loss(
    config: TrainConfig, dataset, device: torch.device
) -> tuple[MultiTaskNowcastNet, MultiTaskLoss]:
    """Instantiate the network and fit the rainfall class weights."""
    settings = get_settings()
    net_config = NowcastNetConfig.preset(
        config.preset, use_cha=config.use_cha, model_version=settings.model_version
    )
    net_config.backbone.predict_steps = config.horizon
    model = MultiTaskNowcastNet(net_config).to(device)
    rain_weights = None
    if config.fit_rain_class_weights and len(dataset):
        counts = torch.zeros(len(net_config.rain_classes), dtype=torch.float64)
        for position in range(min(len(dataset), 8)):
            sample = dataset[position]
            counts += torch.bincount(
                sample.rain_class.reshape(-1), minlength=len(net_config.rain_classes)
            ).to(torch.float64)
        rain_weights = class_balanced_weights(counts)
        print(
            "rain class counts:", counts.tolist(),
            "weights:", [round(float(w), 4) for w in rain_weights],
        )
    criterion = MultiTaskLoss(
        weights=LossWeights(kl=config.kl_weight),
        focal_gamma=config.focal_gamma,
        positive_weight=config.pos_weight,
        rain_class_weights=rain_weights,
    )
    return model, criterion


def _provenance(
    config: TrainConfig,
    *,
    grid: GridSpec,
    dataset,
    split: dict[str, Any],
    spec: WindowSpec,
    event_ids: Sequence[str],
    n_parameters: int,
    device: torch.device,
    epochs_completed: int,
) -> dict[str, Any]:
    """Assemble the training-run record stored beside the checkpoint."""
    return {
        "trained": True,
        "is_synthetic": bool(config.is_synthetic),
        "data_source": SYNTHETIC_ATTRIBUTION if config.is_synthetic else "operational observations",
        "disclaimer": EXPERIMENTAL_DISCLAIMER,
        "validation_status": (
            "SYNTHETIC DEMO - agreement with the SIHPS synthetic generator only; "
            "no independent observational validation and no forecasting skill claim."
            if config.is_synthetic
            else "Observational data - independent validation is still required before "
            "any skill claim."
        ),
        "config": asdict(config),
        "dataset": dataset.provenance_summary(),
        "split": split,
        "window": {"seq_len": spec.seq_len, "horizon": spec.horizon, "stride": spec.stride},
        "lead_hours": spec.horizon_hours,
        "grid": grid.to_dict(),
        "event_ids": list(event_ids),
        "n_parameters": int(n_parameters),
        "device": str(device),
        "seed": int(config.seed),
        "epochs_completed": int(epochs_completed),
    }


def train(config: TrainConfig) -> TrainResult:
    """Run a full training job and write the checkpoint plus its report.

    The Phase 8 data gate runs first, on the exact cubes and configuration this
    run will consume, before any model, optimiser or checkpoint exists. A
    blocked run raises :class:`~app.training.gate.TrainingBlocked` and writes
    nothing.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)
    settings = get_settings()
    grid = settings.grid

    logger.info(
        "starting training run",
        extra={"device": str(device), "epochs": config.epochs, "preset": config.preset},
    )
    if config.is_synthetic:
        if config.mode == PRODUCTION:
            # Do not print the demo banner in a production run: the gate below
            # will refuse, and a demo banner here would be misleading.
            print(
                "[PRODUCTION MODE] --mode production was requested, but the loader "
                "supplies generated cubes. The Phase 8 gate will refuse this run."
            )
        else:
            print(
                "[SYNTHETIC DEMO] Training on SIHPS-generated data. This is a pipeline "
                "demonstration only, NOT an operational forecast. No forecasting "
                "accuracy is claimed or has been validated."
            )

    cubes, event_ids, terrain_tensor = load_training_cubes(config, grid)

    # ------------------------------------------------- phase 8.1 data gate
    # Runs on the exact cubes and configuration this run will consume, before
    # any model, optimiser or checkpoint exists. A blocked run therefore leaves
    # no artefact behind.
    gate_report = _run_data_gate(cubes, config)
    if config.mode == SYNTHETIC_DEMO:
        print(gate_report["next_actions"][0])

    spec = WindowSpec(seq_len=config.seq_len, horizon=config.horizon, stride=1)
    shortest = min(cube.n_frames for cube in cubes)
    if spec.total_frames > shortest:
        raise ValueError(
            f"window needs {spec.total_frames} frames but the shortest event has {shortest}"
        )
    test_fraction = max(0.0, 1.0 - config.train_fraction - config.val_fraction)
    datasets, plans = build_datasets(
        cubes,
        spec,
        terrain=terrain_tensor,
        fractions=(config.train_fraction, config.val_fraction, test_fraction),
        embargo_frames=config.embargo_frames,
        is_synthetic=config.is_synthetic,
        event_ids=event_ids,
    )
    loaders = build_dataloaders(
        datasets, batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed
    )
    if "train" not in loaders:
        raise ValueError("the training split is empty; reduce the embargo or add more events")
    split = split_summary(datasets, plans)
    for name in ("val", "test"):
        if name not in loaders:
            # Silence here would hide the fact that no validation was performed,
            # which matters when a run is later used to justify a checkpoint.
            logger.warning(
                "split is empty after embargo; reduce --embargo-frames or add events",
                extra={"split": name, "embargo_frames": config.embargo_frames},
            )
            print(
                f"[WARNING] the {name} split is empty after the embargo "
                f"(embargo_frames={config.embargo_frames}). No {name} metrics will be produced; "
                f"lower --embargo-frames or train on longer sequences."
            )
    if config.log_every:
        print("split:", json.dumps(split, indent=2, default=str))

    model, criterion = _build_model_and_loss(config, datasets["train"], device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    history, best_val, best_state = _fit(model, loaders, criterion, device, optimiser, config)
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best validation weights", extra={"best_val_loss": best_val})

    provenance = _provenance(
        config,
        grid=grid,
        dataset=datasets["train"],
        split=split,
        spec=spec,
        event_ids=event_ids,
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        device=device,
        epochs_completed=len(history),
    )
    metrics, calibration = _evaluate(model, loaders, device, spec)
    output_dir = Path(config.output_dir)
    save_checkpoint(model, output_dir, metrics=metrics, calibration=calibration, extra=provenance)
    result = TrainResult(
        output_dir=str(output_dir),
        epochs=len(history),
        n_parameters=provenance["n_parameters"],
        best_val_loss=best_val,
        history=history,
        split=split,
        validation_metrics=metrics,
        calibration=calibration,
        provenance=provenance,
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    logger.info("training finished", extra={"output_dir": str(output_dir), "best_val_loss": best_val})
    return result


def _resolve_cape_status(status: str | None) -> str | None:
    """Resolve the configured CAPE validation status.

    ``"run"`` executes the Phase 8 CAPE validator now, so a production run
    cannot claim a status it never computed. A real status string passes through
    unchanged; ``None`` means "not validated" and stays ``None`` so readiness
    blocks it.
    """
    if status != "run":
        return status
    from app.ingestion.realtime.validation import run_cape_validation

    return run_cape_validation().status


def _resolve_iwv_status(status: str | None) -> str | None:
    """Resolve the configured IWV validation status; see :func:`_resolve_cape_status`."""
    if status != "run":
        return status
    from app.ingestion.realtime.validation import run_iwv_validation

    return run_iwv_validation().status


def _run_data_gate(cubes: list[ObservationCube], config: TrainConfig) -> dict[str, Any]:
    """Apply the Phase 8 gate to this run's cubes, configuration and output path.

    Raises :class:`~app.training.gate.TrainingBlocked` with a full, actionable
    report. Called before any model, optimiser or checkpoint is created.
    """
    output_dir = config.output_dir
    if config.mode == SYNTHETIC_DEMO and "synthetic-demo" not in output_dir:
        # Synthetic demonstrations may never write to a production path.
        output_dir = synthetic_demo_output_dir(output_dir)

    cape_status = _resolve_cape_status(config.cape_validation)
    iwv_status = _resolve_iwv_status(config.iwv_validation)
    return enforce_training_gate(
        cubes,
        mode=config.mode,
        output_dir=output_dir,
        seq_len=config.seq_len,
        frame_minutes=FRAME_MINUTES,
        cape_validation=cape_status,
        iwv_validation=iwv_status,
    )


def _fit(
    model: MultiTaskNowcastNet,
    loaders: dict[str, DataLoader],
    criterion: MultiTaskLoss,
    device: torch.device,
    optimiser: torch.optim.Optimizer,
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], float | None, dict[str, torch.Tensor] | None]:
    """Epoch loop; validates each epoch and keeps the best validation weights."""
    history: list[dict[str, Any]] = []
    best_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(max(1, int(config.epochs))):
        train_stats = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimiser=optimiser,
            grad_clip=config.grad_clip,
            max_steps=config.max_steps,
        )
        entry: dict[str, Any] = {"epoch": epoch + 1, "train": train_stats}
        if "val" in loaders and loaders["val"]:
            val_stats = run_epoch(
                model, loaders["val"], criterion, device, max_steps=config.max_steps
            )
            entry["val"] = val_stats
            if best_val is None or val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)
        print(
            f"epoch {entry['epoch']}/{config.epochs} train_loss={train_stats['loss']:.4f}"
            + (f" val_loss={entry['val']['loss']:.4f}" if "val" in entry else "")
        )
    return history, best_val, best_state


def _evaluate(
    model: MultiTaskNowcastNet,
    loaders: dict[str, DataLoader],
    device: torch.device,
    spec: WindowSpec,
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    """Validation metrics + a temperature scaler, both clearly labelled."""
    loader = loaders.get("val") or loaders.get("test")
    if loader is None:
        return None, None
    predictions, targets = collect_predictions(model, loader, device)
    report = evaluate_fields(
        predictions, targets, lead_hours=list(spec.horizon_hours), n_bins=10
    )
    metrics = {"evaluation": report, "summary": summarise(report)}
    print(
        "validation (SYNTHETIC DEMO, not a skill claim): "
        + json.dumps(metrics["summary"], indent=2, default=str)
    )
    truth = targets["cloudburst"].ravel()
    calibration = None
    if truth.size:
        calibration = {"cloudburst": {"temperature": float(TemperatureScaler().fit(predictions["cloudburst"].ravel(), truth))}}
    return metrics, calibration


def _reject_observational(namespace, _values, _option_string=None):
    """Reject ``--observational`` outright (Phase 8.3).

    The flag used to set ``is_synthetic=False`` on whatever the loader produced.
    A deprecation warning would keep a harmful invocation *working* by silently
    ignoring it, so the flag now fails loudly. Use ``--mode production``, which
    verifies provenance and refuses generated data.
    """
    raise SystemExit(
        "error: --observational was removed in Phase 8.3.\n"
        "It previously only flipped a label on generated data. Use --mode production\n"
        "instead, which verifies the dataset's SHA-256 provenance and refuses it."
    )


class _ObservationalRemovedAction(argparse.Action):  # pragma: no cover - parser glue
    def __call__(self, parser, namespace, values, option_string=None):
        _reject_observational(namespace, values, option_string)


def build_parser() -> argparse.ArgumentParser:
    """CLI for ``sihps-train`` / ``python -m app.training.train``."""
    parser = argparse.ArgumentParser(
        prog="sihps-train",
        description=(
            "Train the SIHPS multi-hazard nowcaster. The default dataset is the "
            "clearly-labelled SYNTHETIC demo generator: results demonstrate the "
            "pipeline only and are not an operational forecast."
        ),
    )
    parser.add_argument("--data-dir", default=None, help="Reuse a generated demo dataset from this directory.")
    parser.add_argument("--n-events", type=int, default=4, help="Number of synthetic events to train on.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=6, help="History frames per window.")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast steps per window.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--embargo-frames",
        type=int,
        default=None,
        help="Frames dropped before each val/test boundary. Must be >= --horizon "
        "(default: --horizon).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preset", choices=("lite", "full"), default="lite")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--no-cha", action="store_true", help="Disable Cross-Hazard Attention.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps per epoch (CPU smoke run).")
    parser.add_argument("--output-dir", default="data/models/sihps-convlstm-cha-v0.1.0")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--observational",
        action=_ObservationalRemovedAction,
        nargs=0,
        help=(
            "REMOVED in Phase 8.3. It only flipped a label on generated data. "
            "Use --mode production, which verifies provenance and refuses it."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(PRODUCTION, SYNTHETIC_DEMO),
        default=SYNTHETIC_DEMO,
        help=(
            "production: real data only, fully validated, may write the production "
            f"checkpoint. {SYNTHETIC_DEMO}: a labelled pipeline demonstration that "
            f"may only write beneath a '{DEMO_DIR_MARKER}' directory."
        ),
    )
    parser.add_argument(
        "--cape-validation",
        default="run",
        help="'run' (default) validates now; or pass a Phase 8 status, or 'none'.",
    )
    parser.add_argument(
        "--iwv-validation",
        default="run",
        help="'run' (default) validates now; or pass a Phase 8 status, or 'none'.",
    )
    return parser


def _status_argument(value: str) -> str | None:
    """Map a CLI validation argument onto a config value.

    ``"run"`` means "validate now"; ``"none"`` means "not validated", which
    readiness must treat as a blocker; anything else is passed through as a
    claimed status.
    """
    lowered = (value or "").strip().lower()
    if lowered == "none":
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``sihps-train`` console script."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    config = TrainConfig(
        data_dir=args.data_dir,
        n_events=args.n_events,
        seed=args.seed,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        embargo_frames=args.embargo_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preset=args.preset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
        kl_weight=args.kl_weight,
        use_cha=not args.no_cha,
        device=args.device,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        mode=args.mode,
        cape_validation=_status_argument(args.cape_validation),
        iwv_validation=_status_argument(args.iwv_validation),
    )
    if args.observational:
        # Kept as an explicit, loud refusal rather than a silent relabelling.
        print(
            "[PHASE 8.1] --observational is no longer accepted: it previously only "
            "flipped a label on synthetic data. Production training is now selected "
            "with --mode production, which validates the dataset and refuses "
            "generated inputs."
        )
    try:
        result = train(config)
    except TrainingBlocked as exc:
        print(exc.render())
        return 3
    print(
        json.dumps(
            {
                "output_dir": result.output_dir,
                "epochs": result.epochs,
                "n_parameters": result.n_parameters,
                "best_val_loss": result.best_val_loss,
                "is_synthetic": result.provenance["is_synthetic"],
                "validation_status": result.provenance["validation_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
