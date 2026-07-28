"""Adil, yeniden başlatılabilir Optuna hiperparametre optimizasyonu.

Her trial aynı başlangıç checkpoint'i, seed, eğitim bütçesi ve validation
evaluation manifestini kullanır. Test split'i yalnız ``--run-final`` ile,
parametre seçimi tamamlandıktan sonra değerlendirilir.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import evaluate as evaluation_module
from . import train as training_module
from .backend import resolve_backends
from .bootstrap import PROJECT_ROOT


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Tuning için Optuna gerekli. Aktif PyTorch ortamında "
            "`python -m pip install -r mosaic_system/requirements-tuning.txt` "
            "komutunu çalıştırın."
        ) from exc
    return optuna


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Virgülle ayrılmış tam sayılar gerekli") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("En az bir pozitif tam sayı gerekli")
    return parsed


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Virgülle ayrılmış sayılar gerekli") from exc
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("En az bir sonlu sayı gerekli")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aynı seed ve aynı validation evaluation protokolüyle Optuna "
            "hiperparametre optimizasyonu"
        )
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--hr-base",
        type=Path,
        default=PROJECT_ROOT / "thermal database" / "thermal_dataset_split",
    )
    parser.add_argument(
        "--lr-base",
        type=Path,
        default=PROJECT_ROOT
        / "thermal database"
        / "thermal_dataset_degraded"
        / "x4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "mosaic_system" / "runs" / "optuna_rtx3060",
    )
    parser.add_argument("--study-name", default="thermal_edsr_mosaic_x4")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epochs-per-trial", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--val-max-samples", type=int, default=32)
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=10,
        help="Her trial sonunda aynı val mozaiklerinde tiled evaluation sayısı.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument(
        "--pruner",
        choices=["none", "median"],
        default="none",
        help="'none': her trial tam bütçeyi alır; 'median': zayıf trial erken kesilebilir.",
    )

    parser.add_argument("--patch-sizes", type=_csv_ints, default=(48, 64, 96))
    parser.add_argument(
        "--edge-weights",
        type=_csv_floats,
        default=(0.0, 0.005, 0.01, 0.02, 0.03),
    )
    parser.add_argument(
        "--mosaic-ratios",
        type=_csv_floats,
        default=(0.30, 0.50, 0.70),
    )
    parser.add_argument("--learning-rate-min", type=float, default=2e-6)
    parser.add_argument("--learning-rate-max", type=float, default=3e-5)
    parser.add_argument(
        "--fixed-post-shuffle-relu",
        choices=["search", "on", "off"],
        default="search",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=8,
        help="Patch değişse de gradient accumulation ile sabit tutulur.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seam-mode", choices=["avoid", "mask"], default="avoid")
    parser.add_argument("--seam-margin-lr", type=int, default=4)
    parser.add_argument(
        "--cache-mode",
        choices=["memory", "rolling_disk"],
        default="memory",
    )
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--preprocess-backend",
        choices=["auto", "cpu", "cupy"],
        default="auto",
    )
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument(
        "--run-final",
        action="store_true",
        help=(
            "Arama sonrasında seçilen ayarla temiz uzun eğitim yapar ve test "
            "split'inde eski/yeni modeli aynı seed ile bir kez karşılaştırır."
        ),
    )
    parser.add_argument("--final-epochs", type=int, default=60)
    parser.add_argument("--final-patience", type=int, default=20)
    parser.add_argument(
        "--final-samples-per-epoch",
        type=int,
        default=0,
        help="0: tekrarsız mümkün olan tüm train örnekleri.",
    )
    parser.add_argument("--final-val-max-samples", type=int, default=97)
    parser.add_argument(
        "--final-test-max-samples",
        type=int,
        default=0,
        help="0: tüm test mozaikleri.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"Başlangıç checkpoint'i bulunamadı: {args.pretrained}")
    if args.trials < 0:
        raise ValueError("--trials negatif olamaz")
    for name in (
        "epochs_per_trial",
        "samples_per_epoch",
        "val_max_samples",
        "eval_max_samples",
        "effective_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} pozitif olmalı")
    if not 0 < args.learning_rate_min <= args.learning_rate_max:
        raise ValueError("Learning-rate aralığı pozitif ve sıralı olmalı")
    if any(not 0 < ratio < 1 for ratio in args.mosaic_ratios):
        raise ValueError("Mosaic oranları 0 ile 1 arasında olmalı")
    if args.seam_mode == "avoid":
        max_height = 128 - 2 * args.seam_margin_lr
        if any(size > max_height for size in args.patch_sizes):
            raise ValueError(
                "seam-mode=avoid için patch, 128 - 2×seam-margin değerini "
                f"aşamaz (şu an sınır {max_height})."
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(evaluation_dir: Path) -> str:
    path = evaluation_dir / "evaluated_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation manifesti bulunamadı: {path}")
    return _sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clean_namespace(namespace: argparse.Namespace) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        return value

    return {key: clean(value) for key, value in vars(namespace).items()}


def _training_args(
    tune_args: argparse.Namespace,
    *,
    output_dir: Path,
    patch_size: int,
    learning_rate: float,
    edge_weight: float,
    mosaic_ratio: float,
    post_shuffle_relu: bool,
    epochs: int,
    samples_per_epoch: int,
    val_max_samples: int,
    patience: int,
) -> argparse.Namespace:
    args = training_module.parse_args([])
    args.hr_base = tune_args.hr_base
    args.lr_base = tune_args.lr_base
    args.pretrained = tune_args.pretrained
    args.from_scratch = False
    args.resume = None
    args.output_dir = output_dir
    args.epochs = epochs
    args.patch_size = patch_size
    args.learning_rate = learning_rate
    args.edge_weight = edge_weight
    args.weight_decay = tune_args.weight_decay
    args.gradient_clip = tune_args.gradient_clip
    args.num_workers = tune_args.num_workers
    args.seed = tune_args.seed
    args.deterministic = True
    args.patience = patience
    args.val_max_samples = val_max_samples
    args.samples_per_epoch = samples_per_epoch
    args.paired_ratio = 1.0 - mosaic_ratio
    args.mosaic_ratio = mosaic_ratio
    args.seam_mode = tune_args.seam_mode
    args.seam_margin_lr = tune_args.seam_margin_lr
    args.cache_mode = tune_args.cache_mode
    args.cache_size = tune_args.cache_size
    args.device = tune_args.device
    args.preprocess_backend = tune_args.preprocess_backend
    args.no_amp = tune_args.no_amp
    args.no_post_shuffle_relu = not post_shuffle_relu

    backend = resolve_backends(
        device_request=args.device,
        preprocess_request=args.preprocess_backend,
    )
    automatic = training_module._auto_batch_size(backend.device, patch_size)
    args.batch_size = min(tune_args.effective_batch_size, automatic)
    args.gradient_accumulation = math.ceil(
        tune_args.effective_batch_size / args.batch_size
    )
    return args


def _evaluation_args(
    tune_args: argparse.Namespace,
    *,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    max_samples: int,
    save_previews: int,
) -> argparse.Namespace:
    args = evaluation_module.parse_args([])
    args.checkpoint = checkpoint
    args.hr_base = tune_args.hr_base
    args.hr_dir = None
    args.split = split
    args.output_dir = output_dir
    args.max_samples = max_samples
    args.save_previews = save_previews
    args.seed = tune_args.seed
    args.seam_margin_lr = tune_args.seam_margin_lr
    args.device = tune_args.device
    args.preprocess_backend = tune_args.preprocess_backend
    return args


def _run_evaluation(
    tune_args: argparse.Namespace,
    *,
    checkpoint: Path,
    output_dir: Path,
    split: str = "val",
    max_samples: int | None = None,
    save_previews: int = 0,
) -> dict[str, Any]:
    evaluation_module.run(
        _evaluation_args(
            tune_args,
            checkpoint=checkpoint,
            output_dir=output_dir,
            split=split,
            max_samples=(
                tune_args.eval_max_samples if max_samples is None else max_samples
            ),
            save_previews=save_previews,
        )
    )
    return _read_json(output_dir / "summary.json")


def _ensure_baseline(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], str]:
    output_dir = args.output_dir / "baseline_val_evaluation"
    metadata_path = output_dir / "tuning_baseline.json"
    if (output_dir / "summary.json").is_file() and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        expected = {
            "checkpoint_sha256": checkpoint_sha256,
            "seed": args.seed,
            "max_samples": args.eval_max_samples,
            "split": "val",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                "Mevcut baseline farklı checkpoint/seed/protokolle üretilmiş. "
                "Adil karşılaştırma için yeni bir --output-dir kullanın."
            )
        manifest_sha256 = _manifest_sha256(output_dir)
        if metadata.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("Baseline evaluation manifesti sonradan değişmiş.")
        summary = _read_json(output_dir / "summary.json")
        if int(summary.get("samples", -1)) != args.eval_max_samples:
            raise RuntimeError("Baseline evaluation örnek bütçesi uyuşmuyor.")
        return summary, manifest_sha256

    summary = _run_evaluation(
        args,
        checkpoint=args.pretrained,
        output_dir=output_dir,
    )
    _write_json(
        metadata_path,
        {
            "checkpoint_sha256": checkpoint_sha256,
            "seed": args.seed,
            "max_samples": args.eval_max_samples,
            "split": "val",
            "manifest_sha256": _manifest_sha256(output_dir),
        },
    )
    return summary, _manifest_sha256(output_dir)


def _post_shuffle_choice(args: argparse.Namespace, trial) -> bool:
    if args.fixed_post_shuffle_relu == "on":
        return True
    if args.fixed_post_shuffle_relu == "off":
        return False
    return bool(trial.suggest_categorical("post_shuffle_relu", [True, False]))


def _eligibility(
    summary: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(summary["ssim_model"]) < float(baseline["ssim_model"]):
        reasons.append("SSIM baseline altı")
    phase_limit = float(baseline["model_phase_mean_std"]) * 1.10
    if float(summary["model_phase_mean_std"]) > phase_limit:
        reasons.append("phase artefaktı baseline +%10 sınırını aşıyor")
    baseline_clip = float(baseline["model_clip_ratio"])
    clip_limit = max(baseline_clip * 1.10, baseline_clip + 1e-6)
    if float(summary["model_clip_ratio"]) > clip_limit:
        reasons.append("clipping baseline +%10 sınırını aşıyor")
    return not reasons, reasons


def _constraint_values(
    summary: dict[str, Any],
    baseline: dict[str, Any],
) -> list[float]:
    """Optuna için <=0 uygun, >0 ihlal anlamına gelen kalite sınırları."""
    baseline_clip = float(baseline["model_clip_ratio"])
    return [
        float(baseline["ssim_model"]) - float(summary["ssim_model"]),
        float(summary["model_phase_mean_std"])
        - float(baseline["model_phase_mean_std"]) * 1.10,
        float(summary["model_clip_ratio"])
        - max(baseline_clip * 1.10, baseline_clip + 1e-6),
    ]


def _constraints_func(frozen_trial) -> list[float]:
    return list(frozen_trial.user_attrs.get("quality_constraints", (0.0, 0.0, 0.0)))


def _study_rows(study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "objective_psnr": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"metric_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return rows


def _write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _completed_trials(study) -> list[Any]:
    optuna = _import_optuna()
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]


def _recommended_trial(study):
    completed = _completed_trials(study)
    if not completed:
        raise RuntimeError("Tamamlanmış trial yok; seçim yapılamıyor.")
    eligible = [trial for trial in completed if trial.user_attrs.get("eligible")]
    candidates = eligible or completed
    return max(candidates, key=lambda trial: float(trial.value))


def _plot_study(study, output_dir: Path) -> None:
    completed = sorted(_completed_trials(study), key=lambda trial: trial.number)
    if not completed:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    numbers = [trial.number for trial in completed]
    values = [float(trial.value) for trial in completed]
    best_so_far: list[float] = []
    running = -math.inf
    for value in values:
        running = max(running, value)
        best_so_far.append(running)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.scatter(numbers, values, label="Trial evaluation PSNR", alpha=0.75)
    axis.plot(numbers, best_so_far, label="O ana kadarki en iyi", linewidth=2)
    axis.set_xlabel("Trial")
    axis.set_ylabel("Validation tiled evaluation PSNR (dB)")
    axis.set_title("Optimizasyon geçmişi")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "optimization_history.png", dpi=160)
    plt.close(figure)

    psnr_deltas = [
        float(trial.user_attrs.get("psnr_vs_starting_model", float("nan")))
        for trial in completed
    ]
    ssim_deltas = [
        float(trial.user_attrs.get("ssim_vs_starting_model", float("nan")))
        for trial in completed
    ]
    figure, (psnr_axis, ssim_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )
    colors = [
        "#15803d" if trial.user_attrs.get("eligible") else "#b45309"
        for trial in completed
    ]
    psnr_axis.bar(numbers, psnr_deltas, color=colors)
    psnr_axis.axhline(0.0, color="#555555", linestyle="--")
    psnr_axis.set_ylabel("Yeni − başlangıç (dB)")
    psnr_axis.set_title("Aynı-manifest PSNR farkı")
    psnr_axis.grid(axis="y", alpha=0.25)
    ssim_axis.bar(numbers, ssim_deltas, color=colors)
    ssim_axis.axhline(0.0, color="#555555", linestyle="--")
    ssim_axis.set_xlabel("Trial")
    ssim_axis.set_ylabel("Yeni − başlangıç")
    ssim_axis.set_title("Aynı-manifest SSIM farkı")
    ssim_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "trial_vs_starting_model.png", dpi=160)
    plt.close(figure)

    constraint_labels = ("SSIM", "Phase", "Clipping")
    mean_violations = []
    for index in range(3):
        values_for_constraint = [
            max(
                0.0,
                float(
                    trial.user_attrs.get(
                        "quality_constraints", (0.0, 0.0, 0.0)
                    )[index]
                ),
            )
            for trial in completed
        ]
        mean_violations.append(
            sum(values_for_constraint) / len(values_for_constraint)
        )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(constraint_labels, mean_violations, color="#7c3aed")
    axis.set_title("Trial başına ortalama kalite sınırı ihlali")
    axis.set_ylabel("Pozitif değer = sınır aşımı")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "quality_constraint_violations.png", dpi=160)
    plt.close(figure)

    if len(completed) < 5:
        return
    try:
        optuna = _import_optuna()
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        return
    if not importances:
        return
    names = list(importances)
    scores = [importances[name] for name in names]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(names[::-1], scores[::-1], color="#1464a0")
    axis.set_xlabel("Göreli önem")
    axis.set_title("Hiperparametre önemi")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "parameter_importance.png", dpi=160)
    plt.close(figure)


def _write_recommendation(
    args: argparse.Namespace,
    study,
    baseline: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    trial = _recommended_trial(study)
    params = dict(trial.params)
    if "post_shuffle_relu" not in params:
        params["post_shuffle_relu"] = args.fixed_post_shuffle_relu != "off"
    payload = {
        "study_name": args.study_name,
        "trial_number": trial.number,
        "selection_rule": (
            "Önce SSIM/phase/clipping sınırlarını geçen trial'lar; sonra en yüksek "
            "aynı-manifest tiled validation PSNR."
        ),
        "eligible": bool(trial.user_attrs.get("eligible")),
        "objective_psnr": trial.value,
        "parameters": params,
        "metrics": dict(trial.user_attrs),
        "baseline_metrics": {
            key: baseline.get(key)
            for key in (
                "psnr_model",
                "ssim_model",
                "model_phase_mean_std",
                "model_clip_ratio",
            )
        },
        "fairness": {
            "seed": args.seed,
            "sampler_seed": args.sampler_seed,
            "starting_checkpoint": str(args.pretrained.resolve()),
            "starting_checkpoint_sha256": checkpoint_sha256,
            "epochs_per_trial": args.epochs_per_trial,
            "samples_per_epoch": args.samples_per_epoch,
            "val_max_samples": args.val_max_samples,
            "eval_max_samples": args.eval_max_samples,
            "test_used_for_tuning": False,
            "deterministic_algorithms": True,
        },
    }
    _write_json(args.output_dir / "best_config.json", payload)
    return payload


def _compare_evaluations(
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    def rows(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return {
                row["group_id"]: row
                for row in csv.DictReader(handle)
                if row.get("group_id")
            }

    baseline = rows(baseline_dir / "metrics.csv")
    candidate = rows(candidate_dir / "metrics.csv")
    if set(baseline) != set(candidate):
        raise RuntimeError("Baseline ve yeni model evaluation grupları eşleşmiyor.")
    psnr_deltas = [
        float(candidate[group]["psnr_model"]) - float(baseline[group]["psnr_model"])
        for group in sorted(baseline)
    ]
    ssim_deltas = [
        float(candidate[group]["ssim_model"]) - float(baseline[group]["ssim_model"])
        for group in sorted(baseline)
    ]
    return {
        "groups": len(psnr_deltas),
        "manifest_sha256": _manifest_sha256(candidate_dir),
        "baseline_manifest_sha256": _manifest_sha256(baseline_dir),
        "same_manifest": (
            _manifest_sha256(candidate_dir) == _manifest_sha256(baseline_dir)
        ),
        "mean_psnr_delta_db": sum(psnr_deltas) / len(psnr_deltas),
        "mean_ssim_delta": sum(ssim_deltas) / len(ssim_deltas),
        "psnr_wins": sum(delta > 0 for delta in psnr_deltas),
        "ssim_wins": sum(delta > 0 for delta in ssim_deltas),
    }


def _run_final(
    args: argparse.Namespace,
    recommendation: dict[str, Any],
) -> None:
    params = recommendation["parameters"]
    final_dir = args.output_dir / "final_model"
    if (final_dir / "training_log.csv").exists():
        raise RuntimeError(
            f"Final eğitim klasörü zaten dolu: {final_dir}. Üzerine yazılmadı."
        )
    train_args = _training_args(
        args,
        output_dir=final_dir,
        patch_size=int(params["patch_size"]),
        learning_rate=float(params["learning_rate"]),
        edge_weight=float(params["edge_weight"]),
        mosaic_ratio=float(params["mosaic_ratio"]),
        post_shuffle_relu=bool(params["post_shuffle_relu"]),
        epochs=args.final_epochs,
        samples_per_epoch=args.final_samples_per_epoch,
        val_max_samples=args.final_val_max_samples,
        patience=args.final_patience,
    )
    training_module.run(train_args)

    baseline_dir = args.output_dir / "baseline_test_evaluation"
    candidate_dir = final_dir / "test_evaluation"
    _run_evaluation(
        args,
        checkpoint=args.pretrained,
        output_dir=baseline_dir,
        split="test",
        max_samples=args.final_test_max_samples,
        save_previews=0,
    )
    _run_evaluation(
        args,
        checkpoint=final_dir / "best_model.pth",
        output_dir=candidate_dir,
        split="test",
        max_samples=args.final_test_max_samples,
        save_previews=6,
    )
    comparison = _compare_evaluations(baseline_dir, candidate_dir)
    comparison["seed"] = args.seed
    comparison["split"] = "test"
    comparison["parameter_selection_used_test"] = False
    _write_json(args.output_dir / "final_test_comparison.json", comparison)


def _write_report(
    args: argparse.Namespace,
    study,
    recommendation: dict[str, Any],
) -> None:
    completed = sorted(
        _completed_trials(study),
        key=lambda trial: float(trial.value),
        reverse=True,
    )
    lines = [
        "# Optuna Hiperparametre Optimizasyon Raporu",
        "",
        "## Adalet ve tekrar üretilebilirlik",
        "",
        f"- Bütün trial seed'i: `{args.seed}`",
        f"- Optuna sampler seed'i: `{args.sampler_seed}`",
        f"- Her trial başlangıcı: `{args.pretrained.resolve()}`",
        f"- Her trial bütçesi: `{args.epochs_per_trial}` epoch",
        f"- Trial evaluation split'i: `val`",
        f"- Trial evaluation örneği: `{args.eval_max_samples}`",
        "- Her trial evaluation manifesti baseline manifestiyle SHA-256 olarak doğrulandı.",
        "- Test split'i hiperparametre seçimine dahil edilmedi.",
        "",
        "## Seçilen yapılandırma",
        "",
        f"- Trial: `{recommendation['trial_number']}`",
        f"- Evaluation PSNR: `{recommendation['objective_psnr']:.5f} dB`",
        f"- Kalite sınırlarına uygun: `{recommendation['eligible']}`",
        "",
        "```json",
        json.dumps(recommendation["parameters"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## En iyi trial'lar",
        "",
        "| Trial | Eval PSNR | Eval SSIM | Uygun | Patch | LR | Edge | Mosaic | ReLU |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for trial in completed[:10]:
        params = trial.params
        lines.append(
            f"| {trial.number} | {float(trial.value):.5f} | "
            f"{float(trial.user_attrs['ssim_model']):.6f} | "
            f"{'evet' if trial.user_attrs.get('eligible') else 'hayır'} | "
            f"{params.get('patch_size')} | "
            f"{float(params.get('learning_rate')):.3e} | "
            f"{params.get('edge_weight')} | {params.get('mosaic_ratio')} | "
            f"{params.get('post_shuffle_relu', args.fixed_post_shuffle_relu)} |"
        )
    lines.extend(
        [
            "",
            "## Dosyalar",
            "",
            "- `study.db`: kesintiden sonra devam edebilen Optuna çalışması",
            "- `trials.csv`: bütün parametreler ve evaluation metrikleri",
            "- `best_config.json`: final eğitim için seçilen ayarlar",
            "- `optimization_history.png`: trial sonuçları",
            "- `trial_vs_starting_model.png`: başlangıç modele karşı PSNR/SSIM",
            "- `quality_constraint_violations.png`: kalite sınırı ihlalleri",
            "- `parameter_importance.png`: yeterli trial varsa parametre önemi",
            "- `final_test_comparison.json`: yalnız `--run-final` sonrasında oluşur",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    optuna = _import_optuna()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256(args.pretrained)
    config_path = args.output_dir / "tuning_config.json"
    config = _clean_namespace(args)
    config.update(
        {
            "starting_checkpoint_sha256": checkpoint_sha256,
            "trial_split": "val",
            "test_used_for_tuning": False,
            "deterministic_algorithms": True,
        }
    )
    if config_path.is_file():
        previous = _read_json(config_path)
        protected = (
            "pretrained",
            "starting_checkpoint_sha256",
            "seed",
            "sampler_seed",
            "epochs_per_trial",
            "samples_per_epoch",
            "val_max_samples",
            "eval_max_samples",
            "pruner",
            "patch_sizes",
            "edge_weights",
            "mosaic_ratios",
            "learning_rate_min",
            "learning_rate_max",
            "fixed_post_shuffle_relu",
            "effective_batch_size",
            "weight_decay",
            "gradient_clip",
            "seam_mode",
            "seam_margin_lr",
        )
        if any(previous.get(key) != config.get(key) for key in protected):
            raise RuntimeError(
                "Mevcut study'nin adalet açısından sabit kalması gereken ayarları "
                "değişmiş. Yeni bir --output-dir kullanın."
            )
    else:
        _write_json(config_path, config)

    baseline, baseline_manifest = _ensure_baseline(
        args,
        checkpoint_sha256=checkpoint_sha256,
    )
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        constraints_func=_constraints_func,
    )
    pruner = (
        optuna.pruners.NopPruner()
        if args.pruner == "none"
        else optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
    )
    storage = f"sqlite:///{(args.output_dir / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    # Optuna sampler RNG durumu SQLite'a yazılmaz. Yeniden başlatmada aynı ilk
    # önerileri üretmemesi için seed'i mevcut trial sayısıyla ilerlet.
    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            seed=args.sampler_seed + len(study.trials),
            constraints_func=_constraints_func,
        ),
        pruner=pruner,
    )

    def objective(trial):
        patch_size = int(trial.suggest_categorical("patch_size", args.patch_sizes))
        learning_rate = float(
            trial.suggest_float(
                "learning_rate",
                args.learning_rate_min,
                args.learning_rate_max,
                log=True,
            )
        )
        edge_weight = float(
            trial.suggest_categorical("edge_weight", args.edge_weights)
        )
        mosaic_ratio = float(
            trial.suggest_categorical("mosaic_ratio", args.mosaic_ratios)
        )
        post_shuffle_relu = _post_shuffle_choice(args, trial)
        trial_dir = args.output_dir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        train_args = _training_args(
            args,
            output_dir=trial_dir,
            patch_size=patch_size,
            learning_rate=learning_rate,
            edge_weight=edge_weight,
            mosaic_ratio=mosaic_ratio,
            post_shuffle_relu=post_shuffle_relu,
            epochs=args.epochs_per_trial,
            samples_per_epoch=args.samples_per_epoch,
            val_max_samples=args.val_max_samples,
            patience=args.epochs_per_trial + 1,
        )
        trial.set_user_attr("seed", args.seed)
        trial.set_user_attr("batch_size", train_args.batch_size)
        trial.set_user_attr(
            "gradient_accumulation", train_args.gradient_accumulation
        )

        def epoch_callback(
            epoch: int,
            _train_metrics: dict[str, float],
            val_metrics: dict[str, float],
        ) -> None:
            trial.report(float(val_metrics["psnr"]), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Median pruner trial'i epoch {epoch} sonunda kesti."
                )

        try:
            training_module.run(train_args, epoch_callback=epoch_callback)
            evaluation_dir = trial_dir / "evaluation"
            summary = _run_evaluation(
                args,
                checkpoint=trial_dir / "best_model.pth",
                output_dir=evaluation_dir,
            )
            manifest = _manifest_sha256(evaluation_dir)
            if manifest != baseline_manifest:
                raise RuntimeError(
                    "Trial evaluation manifesti baseline ile eşleşmedi; "
                    "karşılaştırma adil değil."
                )
            eligible, reasons = _eligibility(summary, baseline)
            trial.set_user_attr(
                "quality_constraints",
                _constraint_values(summary, baseline),
            )
            for key in (
                "psnr_model",
                "psnr_bicubic",
                "psnr_gain",
                "ssim_model",
                "ssim_bicubic",
                "ssim_gain",
                "model_clip_ratio",
                "model_phase_mean_std",
                "model_gradient_x",
            ):
                value = summary.get(key)
                if value is not None:
                    trial.set_user_attr(key, float(value))
            trial.set_user_attr(
                "psnr_vs_starting_model",
                float(summary["psnr_model"]) - float(baseline["psnr_model"]),
            )
            trial.set_user_attr(
                "ssim_vs_starting_model",
                float(summary["ssim_model"]) - float(baseline["ssim_model"]),
            )
            trial.set_user_attr("eligible", eligible)
            trial.set_user_attr("ineligibility_reasons", reasons)
            trial.set_user_attr("manifest_sha256", manifest)
            return float(summary["psnr_model"])
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    remaining_trials = max(0, args.trials - len(study.trials))
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)

    rows = _study_rows(study)
    _write_trials_csv(args.output_dir / "trials.csv", rows)
    recommendation = _write_recommendation(
        args,
        study,
        baseline,
        checkpoint_sha256,
    )
    _plot_study(study, args.output_dir)
    _write_report(args, study, recommendation)
    if args.run_final:
        _run_final(args, recommendation)
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"Tuning raporu: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
