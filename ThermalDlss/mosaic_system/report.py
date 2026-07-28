"""Training/evaluation CSV dosyalarından sayısal özet ve PNG grafikler üret."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from .bootstrap import PROJECT_ROOT


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Grafik üretimi için matplotlib gerekli. PyTorch ortamınıza "
            "`matplotlib` kurun."
        ) from exc
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numbers(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _best_row(
    rows: list[dict[str, str]], key: str, maximize: bool = True
) -> dict[str, str] | None:
    valid = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append((value, row))
    if not valid:
        return None
    return (max(valid) if maximize else min(valid))[1]


def plot_training(
    rows: list[dict[str, str]], output_dir: Path
) -> dict[str, float | int | None]:
    if not rows:
        return {}
    plt = _load_matplotlib()
    epochs = _numbers(rows, "epoch")
    train_loss = _numbers(rows, "train_loss")
    val_loss = _numbers(rows, "val_loss")
    val_psnr = _numbers(rows, "val_psnr")
    val_ssim = _numbers(rows, "val_ssim")
    learning_rate = _numbers(rows, "learning_rate")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(epochs[: len(train_loss)], train_loss, label="Train loss", linewidth=2)
    axis.plot(epochs[: len(val_loss)], val_loss, label="Validation loss", linewidth=2)
    axis.set_title("Fine-tuning Kayıp Eğrileri")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_losses.png", dpi=160)
    plt.close(figure)

    figure, left = plt.subplots(figsize=(10, 5.5))
    right = left.twinx()
    psnr_line = left.plot(
        epochs[: len(val_psnr)],
        val_psnr,
        color="#1464a0",
        label="Validation PSNR",
        linewidth=2,
    )
    ssim_line = right.plot(
        epochs[: len(val_ssim)],
        val_ssim,
        color="#d97706",
        label="Validation SSIM",
        linewidth=2,
    )
    left.set_title("Validation Kalitesi")
    left.set_xlabel("Epoch")
    left.set_ylabel("PSNR (dB)", color="#1464a0")
    right.set_ylabel("SSIM", color="#d97706")
    left.grid(alpha=0.25)
    lines = psnr_line + ssim_line
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_psnr_ssim.png", dpi=160)
    plt.close(figure)

    if learning_rate:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(
            epochs[: len(learning_rate)],
            learning_rate,
            color="#7c3aed",
            linewidth=2,
        )
        axis.set_title("Öğrenme Oranı")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "learning_rate.png", dpi=160)
        plt.close(figure)

    best = _best_row(rows, "val_psnr")
    last = rows[-1]
    return {
        "epochs_logged": len(rows),
        "last_epoch": int(float(last["epoch"])),
        "last_train_loss": float(last["train_loss"]),
        "last_val_psnr": float(last["val_psnr"]),
        "last_val_ssim": float(last["val_ssim"]),
        "best_epoch": int(float(best["epoch"])) if best else None,
        "best_val_psnr": float(best["val_psnr"]) if best else None,
        "best_val_ssim": float(best["val_ssim"]) if best else None,
    }


def _rows_by_group(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["group_id"]: row for row in rows if row.get("group_id")}


def plot_evaluation(
    rows: list[dict[str, str]],
    output_dir: Path,
    baseline_rows: list[dict[str, str]] | None = None,
) -> dict[str, float | int | None]:
    if not rows:
        return {}
    plt = _load_matplotlib()
    model_psnr = _numbers(rows, "psnr_model")
    bicubic_psnr = _numbers(rows, "psnr_bicubic")
    gains = _numbers(rows, "psnr_gain")

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(bicubic_psnr, model_psnr, s=25, alpha=0.7)
    bounds = [
        min(bicubic_psnr + model_psnr),
        max(bicubic_psnr + model_psnr),
    ]
    axis.plot(bounds, bounds, linestyle="--", color="#555555", label="Eşitlik")
    axis.set_title("EDSR ve Bicubic PSNR")
    axis.set_xlabel("Bicubic PSNR (dB)")
    axis.set_ylabel("EDSR PSNR (dB)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_psnr_scatter.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(gains, bins=min(20, max(5, len(gains) // 3)), color="#1464a0")
    axis.axvline(0.0, color="#b91c1c", linestyle="--", label="Bicubic ile eşit")
    axis.axvline(mean(gains), color="#15803d", label=f"Ortalama {mean(gains):+.3f}")
    axis.set_title("Bicubic'e Göre PSNR Kazanç Dağılımı")
    axis.set_xlabel("PSNR kazancı (dB)")
    axis.set_ylabel("Mozaik sayısı")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_psnr_gain_histogram.png", dpi=160)
    plt.close(figure)

    artifact_keys = [
        ("clip_ratio", "Clipping oranı"),
        ("phase_mean_std", "4×4 faz std."),
        ("gradient_x", "Yatay gradyan"),
        ("laplacian_abs", "Laplacian"),
    ]
    model_artifacts = [
        mean(_numbers(rows, f"model_{key}")) for key, _ in artifact_keys
    ]
    bicubic_artifacts = [
        mean(_numbers(rows, f"bicubic_{key}")) for key, _ in artifact_keys
    ]
    x = list(range(len(artifact_keys)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        [value - width / 2 for value in x],
        bicubic_artifacts,
        width,
        label="Bicubic",
    )
    axis.bar(
        [value + width / 2 for value in x],
        model_artifacts,
        width,
        label="EDSR",
    )
    axis.set_xticks(x, [label for _, label in artifact_keys], rotation=12)
    axis.set_title("Ortalama Artefakt / Keskinlik Göstergeleri")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_artifacts.png", dpi=160)
    plt.close(figure)

    baseline_comparison = None
    if baseline_rows:
        current = _rows_by_group(rows)
        baseline = _rows_by_group(baseline_rows)
        common = sorted(set(current) & set(baseline))
        differences = [
            float(current[group]["psnr_model"])
            - float(baseline[group]["psnr_model"])
            for group in common
        ]
        if differences:
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.hist(
                differences,
                bins=min(20, max(5, len(differences) // 3)),
                color="#7c3aed",
            )
            axis.axvline(0.0, color="#b91c1c", linestyle="--")
            axis.axvline(
                mean(differences),
                color="#15803d",
                label=f"Ortalama {mean(differences):+.3f} dB",
            )
            axis.set_title("Yeni Model − Eski Model PSNR")
            axis.set_xlabel("PSNR farkı (dB)")
            axis.set_ylabel("Mozaik sayısı")
            axis.grid(alpha=0.2)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_dir / "new_vs_old_psnr.png", dpi=160)
            plt.close(figure)
            baseline_comparison = {
                "common_groups": len(common),
                "new_minus_old_psnr_mean": mean(differences),
                "new_minus_old_psnr_median": median(differences),
                "new_better_fraction": sum(value > 0 for value in differences)
                / len(differences),
            }

    summary: dict[str, float | int | None] = {
        "samples": len(rows),
        "model_psnr_mean": mean(model_psnr),
        "bicubic_psnr_mean": mean(bicubic_psnr),
        "psnr_gain_mean": mean(gains),
        "psnr_gain_median": median(gains),
        "beats_bicubic_fraction": sum(value > 0 for value in gains) / len(gains),
        "model_ssim_mean": (
            mean(_numbers(rows, "ssim_model"))
            if _numbers(rows, "ssim_model")
            else None
        ),
        "bicubic_ssim_mean": (
            mean(_numbers(rows, "ssim_bicubic"))
            if _numbers(rows, "ssim_bicubic")
            else None
        ),
        "model_clip_ratio": model_artifacts[0],
        "bicubic_clip_ratio": bicubic_artifacts[0],
        "model_phase_mean_std": model_artifacts[1],
        "bicubic_phase_mean_std": bicubic_artifacts[1],
    }
    if baseline_comparison:
        summary.update(baseline_comparison)
    return summary


def _write_markdown(
    path: Path,
    training: dict,
    evaluation: dict,
    *,
    training_csv: Path,
    evaluation_csv: Path,
    baseline_csv: Path | None,
) -> None:
    lines = [
        "# Otomatik Model Sonuç Raporu",
        "",
        f"- Training CSV: `{training_csv}`",
        f"- Evaluation CSV: `{evaluation_csv}`",
    ]
    if baseline_csv:
        lines.append(f"- Baseline CSV: `{baseline_csv}`")
    lines.extend(["", "## Eğitim özeti", ""])
    if training:
        lines.extend(
            [
                f"- Loglanan epoch: {training['epochs_logged']}",
                f"- En iyi epoch: {training['best_epoch']}",
                f"- En iyi validation PSNR: {training['best_val_psnr']:.4f} dB",
                f"- En iyi validation SSIM: {training['best_val_ssim']:.6f}",
                f"- Son train loss: {training['last_train_loss']:.7f}",
            ]
        )
    else:
        lines.append("- Training log bulunamadı.")
    lines.extend(["", "## Evaluation özeti", ""])
    if evaluation:
        lines.extend(
            [
                f"- Örnek: {evaluation['samples']}",
                f"- Model PSNR: {evaluation['model_psnr_mean']:.4f} dB",
                f"- Bicubic PSNR: {evaluation['bicubic_psnr_mean']:.4f} dB",
                f"- Ortalama kazanç: {evaluation['psnr_gain_mean']:+.4f} dB",
                f"- Bicubic'i geçen oran: %{evaluation['beats_bicubic_fraction'] * 100:.2f}",
                f"- Model clipping: %{evaluation['model_clip_ratio'] * 100:.4f}",
                f"- Bicubic clipping: %{evaluation['bicubic_clip_ratio'] * 100:.4f}",
            ]
        )
        if evaluation.get("new_minus_old_psnr_mean") is not None:
            lines.extend(
                [
                    f"- Yeni − eski model PSNR: "
                    f"{evaluation['new_minus_old_psnr_mean']:+.4f} dB",
                    f"- Yeni modelin eskiyi geçtiği oran: "
                    f"%{evaluation['new_better_fraction'] * 100:.2f}",
                ]
            )
    else:
        lines.append("- Evaluation metrics bulunamadı.")
    lines.extend(
        [
            "",
            "## Grafikler",
            "",
            "- `training_losses.png`",
            "- `validation_psnr_ssim.png`",
            "- `learning_rate.png`",
            "- `evaluation_psnr_scatter.png`",
            "- `evaluation_psnr_gain_histogram.png`",
            "- `evaluation_artifacts.png`",
            "- `new_vs_old_psnr.png` (baseline verilirse)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic model grafik/özet raporu")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "mosaic_system" / "runs" / "native_x4",
    )
    parser.add_argument("--training-log", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--baseline-evaluation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    training_csv = args.training_log or args.run_dir / "training_log.csv"
    evaluation_dir = args.evaluation_dir or args.run_dir / "evaluation"
    evaluation_csv = evaluation_dir / "metrics.csv"
    baseline_csv = (
        args.baseline_evaluation_dir / "metrics.csv"
        if args.baseline_evaluation_dir
        else None
    )
    output_dir = args.output_dir or args.run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = _read_csv(training_csv)
    evaluation_rows = _read_csv(evaluation_csv)
    baseline_rows = _read_csv(baseline_csv) if baseline_csv else []
    if not training_rows and not evaluation_rows:
        raise FileNotFoundError(
            f"Training veya evaluation CSV bulunamadı: {training_csv}, "
            f"{evaluation_csv}"
        )

    training_summary = plot_training(training_rows, output_dir)
    evaluation_summary = plot_evaluation(
        evaluation_rows,
        output_dir,
        baseline_rows=baseline_rows,
    )
    combined = {
        "training": training_summary,
        "evaluation": evaluation_summary,
    }
    (output_dir / "report_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        output_dir / "REPORT.md",
        training_summary,
        evaluation_summary,
        training_csv=training_csv,
        evaluation_csv=evaluation_csv,
        baseline_csv=baseline_csv,
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"Rapor: {output_dir.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

