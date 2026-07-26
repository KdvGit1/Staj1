"""Dependency-free SVG and HTML reports for dataset and model outputs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data_utils import TARGET_NAMES, atomic_write_text, load_json

SERIES_COLORS = {
    "person": "#e76f51",
    "bike_motorcycle": "#2a9d8f",
    "car": "#457b9d",
}
GENERAL_COLORS = ["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51", "#7b61a8"]
TEXT = "#1f2937"
MUTED = "#64748b"
GRID = "#d9e0e7"
BACKGROUND = "#ffffff"
PANEL = "#f8fafc"


def _fmt_number(value: float | int) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{int(value):,}".replace(",", ".")


def _fmt_percent(value: float) -> str:
    return f"%{value * 100:.1f}"


def _friendly_metric(name: str) -> str:
    mapping = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "p": "Precision",
        "r": "Recall",
        "ap50": "mAP50",
        "ap": "mAP50-95",
        "map50_95": "mAP50-95",
    }
    return mapping.get(name, name.replace("metrics/", "").replace("(B)", ""))


def _svg_document(
    title: str,
    description: str,
    body: str,
    width: int = 1200,
    height: int = 720,
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title desc">\n'
        f"<title id=\"title\">{escape(title)}</title>\n"
        f"<desc id=\"desc\">{escape(description)}</desc>\n"
        "<style>\n"
        "text { font-family: 'Segoe UI', Arial, sans-serif; fill: #1f2937; }\n"
        ".title { font-size: 28px; font-weight: 600; }\n"
        ".subtitle { font-size: 15px; fill: #64748b; }\n"
        ".axis { font-size: 13px; fill: #64748b; }\n"
        ".value { font-size: 13px; font-weight: 600; }\n"
        ".legend { font-size: 14px; }\n"
        ".grid { stroke: #d9e0e7; stroke-width: 1; }\n"
        ".axis-line { stroke: #94a3b8; stroke-width: 1.2; }\n"
        "</style>\n"
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>\n'
        f"{body}\n</svg>\n"
    )


def _save_svg(
    path: Path,
    title: str,
    description: str,
    body: str,
    width: int = 1200,
    height: int = 720,
) -> Path:
    atomic_write_text(path, _svg_document(title, description, body, width, height))
    return path


def _dataset_overview_svg(manifest: dict[str, Any], destination: Path) -> Path:
    splits = manifest["splits"]
    split_order = ("train", "val", "test")
    max_images = max(int(splits[name]["images"]) for name in split_order)
    max_boxes = max(int(splits[name]["target_boxes"]) for name in split_order)
    body = [
        '<text x="60" y="52" class="title">Veri seti genel görünümü</text>',
        '<text x="60" y="80" class="subtitle">'
        "Etiketli/hedefsiz görüntüler ve hedef kutuları; tüm değerler doğrudan yazılmıştır."
        "</text>",
        f'<rect x="45" y="110" width="535" height="545" rx="14" fill="{PANEL}"/>',
        f'<rect x="620" y="110" width="535" height="545" rx="14" fill="{PANEL}"/>',
        '<text x="75" y="150" font-size="20" font-weight="600">Görüntüler</text>',
        '<text x="650" y="150" font-size="20" font-weight="600">Hedef kutuları</text>',
        '<rect x="75" y="174" width="16" height="16" fill="#457b9d"/>',
        '<text x="98" y="187" class="legend">Hedef içeren</text>',
        '<rect x="220" y="174" width="16" height="16" fill="#e9c46a"/>',
        '<text x="243" y="187" class="legend">Hedefsiz negatif</text>',
    ]
    for index, split in enumerate(split_order):
        row_y = 235 + index * 135
        item = splits[split]
        images = int(item["images"])
        empty = int(item["empty_target_images"])
        labeled = images - empty
        full_width = 390 * images / max_images
        labeled_width = full_width * labeled / max(images, 1)
        empty_width = full_width - labeled_width
        body.extend(
            [
                f'<text x="75" y="{row_y}" font-size="17" font-weight="600">{split}</text>',
                f'<rect x="75" y="{row_y + 18}" width="{labeled_width:.2f}" '
                'height="34" rx="5" fill="#457b9d"/>',
                f'<rect x="{75 + labeled_width:.2f}" y="{row_y + 18}" '
                f'width="{max(empty_width, 1):.2f}" height="34" rx="4" fill="#e9c46a"/>',
                f'<text x="{75 + full_width + 10:.2f}" y="{row_y + 41}" '
                f'class="value">{_fmt_number(images)}</text>',
                f'<text x="75" y="{row_y + 76}" class="axis">'
                f'Hedef içeren: {_fmt_number(labeled)} · Hedefsiz: {_fmt_number(empty)}</text>',
            ]
        )
        boxes = int(item["target_boxes"])
        box_width = 390 * boxes / max_boxes
        body.extend(
            [
                f'<text x="650" y="{row_y}" font-size="17" font-weight="600">{split}</text>',
                f'<rect x="650" y="{row_y + 18}" width="{box_width:.2f}" '
                'height="34" rx="5" fill="#2a9d8f"/>',
                f'<text x="{650 + box_width + 10:.2f}" y="{row_y + 41}" '
                f'class="value">{_fmt_number(boxes)}</text>',
                f'<text x="650" y="{row_y + 76}" class="axis">'
                f'Görüntü başına ortalama: {boxes / max(images, 1):.2f}</text>',
            ]
        )
    return _save_svg(
        destination,
        "Veri seti genel görünümü",
        "Train, validation ve test görüntü sayıları, hedefsiz görüntüler ve hedef kutuları.",
        "\n".join(body),
    )


def _class_distribution_svg(manifest: dict[str, Any], destination: Path) -> Path:
    splits = manifest["splits"]
    split_order = ("train", "val", "test")
    class_order = tuple(TARGET_NAMES.values())
    width, height = 1200, 760
    left, top, plot_width, plot_height = 105, 155, 1035, 500
    maximum = max(
        int(splits[split]["class_counts"][class_name])
        for split in split_order
        for class_name in class_order
    )
    max_log = math.ceil(math.log10(maximum))
    body = [
        '<text x="60" y="52" class="title">Sınıf dağılımı</text>',
        '<text x="60" y="80" class="subtitle">'
        "Nadir sınıfların da görülebilmesi için dikey eksen logaritmiktir; bar üstlerinde gerçek kutu sayısı bulunur."
        "</text>",
    ]
    legend_x = 585
    for index, class_name in enumerate(class_order):
        color = SERIES_COLORS[class_name]
        x = legend_x + index * 185
        body.extend(
            [
                f'<rect x="{x}" y="105" width="16" height="16" fill="{color}"/>',
                f'<text x="{x + 23}" y="118" class="legend">{escape(class_name)}</text>',
            ]
        )
    for exponent in range(0, max_log + 1):
        value = 10**exponent
        y = top + plot_height - (math.log10(value + 1) / math.log10(10**max_log + 1)) * plot_height
        label = _fmt_number(value)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{label}</text>',
            ]
        )
    group_width = plot_width / len(split_order)
    bar_width = 70
    gap = 14
    for split_index, split in enumerate(split_order):
        group_start = left + split_index * group_width
        bars_total = len(class_order) * bar_width + (len(class_order) - 1) * gap
        first_x = group_start + (group_width - bars_total) / 2
        for class_index, class_name in enumerate(class_order):
            value = int(splits[split]["class_counts"][class_name])
            bar_height = (
                math.log10(value + 1) / math.log10(10**max_log + 1)
            ) * plot_height
            x = first_x + class_index * (bar_width + gap)
            y = top + plot_height - bar_height
            body.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
                    f'height="{bar_height:.2f}" rx="5" fill="{SERIES_COLORS[class_name]}"/>',
                    f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 9, 142):.2f}" '
                    f'text-anchor="middle" class="value">{_fmt_number(value)}</text>',
                ]
            )
        body.append(
            f'<text x="{group_start + group_width / 2:.2f}" y="{top + plot_height + 38}" '
            f'text-anchor="middle" font-size="17" font-weight="600">{split}</text>'
        )
    body.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" class="axis-line"/>',
        ]
    )
    return _save_svg(
        destination,
        "Sınıf dağılımı",
        "Her split için person, bike motorcycle ve car kutu sayıları logaritmik eksende.",
        "\n".join(body),
        width,
        height,
    )


def _class_share_svg(manifest: dict[str, Any], destination: Path) -> Path:
    splits = manifest["splits"]
    split_order = ("train", "val", "test")
    class_order = tuple(TARGET_NAMES.values())
    body = [
        '<text x="60" y="52" class="title">Split içindeki sınıf payları</text>',
        '<text x="60" y="80" class="subtitle">'
        "Her çubuk yüzde 100'dür; alt satır tüm yüzdeleri ve gerçek sayıları verir."
        "</text>",
    ]
    for index, class_name in enumerate(class_order):
        x = 505 + index * 205
        body.extend(
            [
                f'<rect x="{x}" y="105" width="16" height="16" '
                f'fill="{SERIES_COLORS[class_name]}"/>',
                f'<text x="{x + 23}" y="118" class="legend">{escape(class_name)}</text>',
            ]
        )
    for split_index, split in enumerate(split_order):
        item = splits[split]
        counts = item["class_counts"]
        total = sum(int(counts[class_name]) for class_name in class_order)
        y = 190 + split_index * 165
        body.append(
            f'<text x="80" y="{y + 29}" font-size="18" font-weight="600">{split}</text>'
        )
        current_x = 185.0
        for class_name in class_order:
            value = int(counts[class_name])
            share = value / max(total, 1)
            segment_width = 900 * share
            body.append(
                f'<rect x="{current_x:.2f}" y="{y}" width="{segment_width:.2f}" '
                f'height="55" fill="{SERIES_COLORS[class_name]}"/>'
            )
            if share >= 0.07:
                body.append(
                    f'<text x="{current_x + segment_width / 2:.2f}" y="{y + 34}" '
                    f'text-anchor="middle" font-size="15" font-weight="600" fill="#ffffff" '
                    f'style="fill:#ffffff">{_fmt_percent(share)}</text>'
                )
            current_x += segment_width
        detail = " · ".join(
            f"{class_name}: {_fmt_number(int(counts[class_name]))} "
            f"({_fmt_percent(int(counts[class_name]) / max(total, 1))})"
            for class_name in class_order
        )
        body.append(f'<text x="185" y="{y + 83}" class="axis">{escape(detail)}</text>')
    return _save_svg(
        destination,
        "Split içindeki sınıf payları",
        "Train, validation ve test bölümlerindeki üç hedef sınıfın yüzdesel payları.",
        "\n".join(body),
        1200,
        700,
    )


def _html_page(title: str, intro: str, sections: Sequence[str]) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="tr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>\n"
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;background:#eef2f6;color:#1f2937}"
        "main{max-width:1240px;margin:0 auto;padding:28px}"
        "h1{margin:0 0 8px;font-size:30px}h2{margin-top:34px}"
        "p{line-height:1.55}.muted{color:#64748b}"
        "section{background:#fff;border:1px solid #d9e0e7;border-radius:14px;"
        "padding:20px;margin:20px 0;box-shadow:0 3px 14px rgba(31,41,55,.05)}"
        "img{display:block;width:100%;height:auto}"
        "table{width:100%;border-collapse:collapse;margin-top:14px}"
        "th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right}"
        "th:first-child,td:first-child{text-align:left}th{background:#f8fafc}"
        "code{background:#f1f5f9;padding:2px 5px;border-radius:4px}"
        "</style>\n</head>\n<body>\n<main>\n"
        f"<h1>{escape(title)}</h1>\n<p class=\"muted\">{escape(intro)}</p>\n"
        + "\n".join(sections)
        + "\n</main>\n</body>\n</html>\n"
    )


def generate_dataset_report(
    manifest_path: Path,
    verification_path: Path | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate complete dataset SVGs, HTML and text summary."""

    manifest = load_json(manifest_path)
    verification = (
        load_json(verification_path)
        if verification_path is not None and verification_path.is_file()
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = _dataset_overview_svg(
        manifest, output_dir / "dataset-overview.svg"
    )
    distribution = _class_distribution_svg(
        manifest, output_dir / "class-distribution.svg"
    )
    shares = _class_share_svg(manifest, output_dir / "class-shares.svg")

    rows: list[str] = []
    totals = {"images": 0, "boxes": 0, "empty": 0}
    for split in ("train", "val", "test"):
        item = manifest["splits"][split]
        totals["images"] += int(item["images"])
        totals["boxes"] += int(item["target_boxes"])
        totals["empty"] += int(item["empty_target_images"])
        counts = item["class_counts"]
        rows.append(
            "<tr>"
            f"<td>{split}</td><td>{_fmt_number(item['images'])}</td>"
            f"<td>{_fmt_number(item['empty_target_images'])}</td>"
            f"<td>{_fmt_number(counts['person'])}</td>"
            f"<td>{_fmt_number(counts['bike_motorcycle'])}</td>"
            f"<td>{_fmt_number(counts['car'])}</td>"
            f"<td>{_fmt_number(item['target_boxes'])}</td>"
            "</tr>"
        )
    match_text = (
        "Evet" if verification and verification.get("manifest_match") is True else "Kontrol edilmedi"
    )
    sections = [
        '<section><img src="dataset-overview.svg" alt="Veri seti genel görünümü"></section>',
        '<section><img src="class-distribution.svg" alt="Sınıf dağılımı"></section>',
        '<section><img src="class-shares.svg" alt="Sınıf yüzdeleri"></section>',
        "<section><h2>Tüm sayısal değerler</h2>"
        "<table><thead><tr><th>Split</th><th>Görüntü</th><th>Hedefsiz</th>"
        "<th>person</th><th>bike_motorcycle</th><th>car</th><th>Toplam kutu</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f"<p>Manifest–bağımsız doğrulama eşleşmesi: <strong>{match_text}</strong>.</p>"
        "</section>",
    ]
    html_path = output_dir / "dataset-report.html"
    atomic_write_text(
        html_path,
        _html_page(
            "Termal veri seti grafik raporu",
            "640×512 termal YOLO26n veri setinin eksiksiz sınıf ve split özeti.",
            sections,
        ),
    )
    summary_path = output_dir / "dataset-report-summary.txt"
    atomic_write_text(
        summary_path,
        "TERMAL VERİ SETİ GRAFİK RAPORU ÖZETİ\n"
        "=====================================\n\n"
        f"Oluşturulma (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Toplam görüntü: {_fmt_number(totals['images'])}\n"
        f"Toplam hedef kutusu: {_fmt_number(totals['boxes'])}\n"
        f"Hedefsiz negatif görüntü: {_fmt_number(totals['empty'])}\n"
        f"Manifest doğrulama eşleşmesi: {match_text}\n\n"
        "Grafikler:\n"
        "  dataset-overview.svg: görüntü, negatif örnek ve kutu sayıları\n"
        "  class-distribution.svg: gerçek sınıf sayıları, logaritmik eksen\n"
        "  class-shares.svg: her split içindeki sınıf yüzdeleri\n"
        "  dataset-report.html: bütün grafikler ve eksiksiz sayısal tablo\n",
    )
    return {
        "overview": overview,
        "distribution": distribution,
        "shares": shares,
        "html": html_path,
        "summary": summary_path,
    }


def _read_results_csv(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No columns in training results: {path}")
        rows = list(reader)
    clean_rows: list[dict[str, float]] = []
    for row in rows:
        clean: dict[str, float] = {}
        for key, raw_value in row.items():
            if key is None or raw_value is None or not raw_value.strip():
                continue
            try:
                clean[key.strip()] = float(raw_value.strip())
            except ValueError:
                continue
        clean_rows.append(clean)
    epoch_key = next((key for key in clean_rows[0] if key.lower() == "epoch"), None)
    epochs = [
        row.get(epoch_key, float(index + 1)) if epoch_key else float(index + 1)
        for index, row in enumerate(clean_rows)
    ]
    columns: dict[str, list[float]] = {}
    all_keys = sorted({key for row in clean_rows for key in row})
    for key in all_keys:
        if key == epoch_key:
            continue
        values = [row.get(key, math.nan) for row in clean_rows]
        if any(math.isfinite(value) for value in values):
            columns[key] = values
    return epochs, columns


def _line_chart_svg(
    title: str,
    subtitle: str,
    x_values: Sequence[float],
    series: dict[str, Sequence[float]],
    destination: Path,
    fixed_y: tuple[float, float] | None = None,
    highlight_index: int | None = None,
) -> Path:
    width, height = 1200, 720
    left, top, plot_width, plot_height = 95, 145, 1040, 480
    finite_values = [
        float(value)
        for values in series.values()
        for value in values
        if math.isfinite(float(value))
    ]
    if not finite_values:
        raise ValueError(f"No finite values for chart {title}")
    y_min, y_max = fixed_y or (min(finite_values), max(finite_values))
    if fixed_y is None:
        padding = max((y_max - y_min) * 0.1, abs(y_max) * 0.03, 1e-6)
        y_min = max(0.0, y_min - padding)
        y_max += padding
    if y_max <= y_min:
        y_max = y_min + 1.0
    x_min, x_max = min(x_values), max(x_values)
    if x_max <= x_min:
        x_max = x_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    body = [
        f'<text x="60" y="52" class="title">{escape(title)}</text>',
        f'<text x="60" y="80" class="subtitle">{escape(subtitle)}</text>',
    ]
    legend_start = max(60, 1140 - len(series) * 180)
    for index, label in enumerate(series):
        x = legend_start + index * 180
        color = GENERAL_COLORS[index % len(GENERAL_COLORS)]
        body.extend(
            [
                f'<line x1="{x}" y1="112" x2="{x + 25}" y2="112" '
                f'stroke="{color}" stroke-width="4"/>',
                f'<text x="{x + 33}" y="117" class="legend">{escape(_friendly_metric(label))}</text>',
            ]
        )
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
                f'class="axis">{value:.3f}</text>',
            ]
        )
    x_tick_count = min(8, len(x_values))
    for tick in range(x_tick_count):
        index = round(tick * (len(x_values) - 1) / max(x_tick_count - 1, 1))
        x = sx(float(x_values[index]))
        body.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 32}" text-anchor="middle" '
            f'class="axis">{_fmt_number(x_values[index])}</text>'
        )
    for series_index, (label, values) in enumerate(series.items()):
        color = GENERAL_COLORS[series_index % len(GENERAL_COLORS)]
        points = [
            (sx(float(x)), sy(float(y)))
            for x, y in zip(x_values, values, strict=True)
            if math.isfinite(float(y))
        ]
        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        body.append(
            f'<polyline points="{path_points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if points:
            last_x, last_y = points[-1]
            last_value = next(
                float(value)
                for value in reversed(values)
                if math.isfinite(float(value))
            )
            body.extend(
                [
                    f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="5" fill="{color}"/>',
                    f'<text x="{min(last_x + 8, 1130):.2f}" y="{last_y - 8:.2f}" '
                    f'class="value">{last_value:.4f}</text>',
                ]
            )
    if highlight_index is not None and 0 <= highlight_index < len(x_values):
        x = sx(float(x_values[highlight_index]))
        body.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
                'stroke="#e76f51" stroke-width="2" stroke-dasharray="7 5"/>',
                f'<text x="{x:.2f}" y="{top - 12}" text-anchor="middle" '
                'font-size="14" font-weight="600">en iyi epoch</text>',
            ]
        )
    body.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" class="axis-line"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 35}" '
            'text-anchor="middle" class="axis">Epoch</text>',
        ]
    )
    return _save_svg(destination, title, subtitle, "\n".join(body), width, height)


def generate_training_report(run_dir: Path) -> dict[str, Path]:
    """Generate training charts from an Ultralytics ``results.csv``."""

    run_dir = run_dir.resolve()
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training results.csv not found: {csv_path}")
    epochs, columns = _read_results_csv(csv_path)
    output_dir = run_dir / "graph-report"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    train_losses = {
        key: values for key, values in columns.items() if key.startswith("train/") and "loss" in key
    }
    val_losses = {
        key: values for key, values in columns.items() if key.startswith("val/") and "loss" in key
    }
    metrics = {
        key: values for key, values in columns.items() if key.startswith("metrics/")
    }
    learning_rates = {
        key: values for key, values in columns.items() if key.startswith("lr/")
    }
    time_columns = {
        key: values for key, values in columns.items() if key.lower() == "time"
    }
    map_key = next((key for key in metrics if "mAP50-95" in key), None)
    best_index = None
    if map_key:
        finite_pairs = [
            (index, float(value))
            for index, value in enumerate(metrics[map_key])
            if math.isfinite(float(value))
        ]
        if finite_pairs:
            best_index = max(finite_pairs, key=lambda pair: pair[1])[0]
    if train_losses:
        outputs["train_losses"] = _line_chart_svg(
            "Eğitim loss eğrileri",
            "Düşüş beklenir; ani sıçramalar veri veya öğrenme oranı sorununa işaret edebilir.",
            epochs,
            train_losses,
            output_dir / "training-losses.svg",
            highlight_index=best_index,
        )
    if val_losses:
        outputs["val_losses"] = _line_chart_svg(
            "Validation loss eğrileri",
            "Train düşerken validation yükseliyorsa overfitting ihtimali vardır.",
            epochs,
            val_losses,
            output_dir / "validation-losses.svg",
            highlight_index=best_index,
        )
    if metrics:
        outputs["metrics"] = _line_chart_svg(
            "Validation tespit metrikleri",
            "Precision, recall, mAP50 ve mAP50-95 değerlerinin epoch boyunca değişimi.",
            epochs,
            metrics,
            output_dir / "validation-metrics.svg",
            fixed_y=(0.0, 1.0),
            highlight_index=best_index,
        )
    if learning_rates:
        outputs["learning_rates"] = _line_chart_svg(
            "Öğrenme oranı",
            "Optimizer parametre gruplarının epoch bazındaki learning-rate programı.",
            epochs,
            learning_rates,
            output_dir / "learning-rate.svg",
            highlight_index=best_index,
        )
    if time_columns:
        outputs["time"] = _line_chart_svg(
            "Kümülatif eğitim süresi",
            "Epoch tamamlandıkça biriken çalışma süresi; results.csv içindeki time sütunu.",
            epochs,
            time_columns,
            output_dir / "training-time.svg",
            highlight_index=best_index,
        )
    best_epoch = epochs[best_index] if best_index is not None else None
    chart_sections = [
        f'<section><img src="{path.name}" alt="{escape(key)}"></section>'
        for key, path in outputs.items()
    ]
    table_rows = []
    for key, values in columns.items():
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            continue
        table_rows.append(
            f"<tr><td>{escape(_friendly_metric(key))}</td>"
            f"<td>{finite[-1]:.6f}</td><td>{min(finite):.6f}</td>"
            f"<td>{max(finite):.6f}</td></tr>"
        )
    chart_sections.append(
        "<section><h2>Tüm CSV sütunlarının özeti</h2>"
        "<table><thead><tr><th>Metrik</th><th>Son</th><th>Minimum</th><th>Maksimum</th>"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></section>"
    )
    html_path = output_dir / "training-report.html"
    atomic_write_text(
        html_path,
        _html_page(
            "YOLO26n eğitim grafik raporu",
            f"Run: {run_dir.name}. En iyi epoch: "
            + (_fmt_number(best_epoch) if best_epoch is not None else "belirlenemedi"),
            chart_sections,
        ),
    )
    summary_path = output_dir / "training-report-summary.txt"
    atomic_write_text(
        summary_path,
        "YOLO26N EĞİTİM GRAFİK RAPORU\n"
        "=============================\n\n"
        f"Run: {run_dir}\n"
        f"Epoch sayısı: {len(epochs)}\n"
        f"En iyi epoch (mAP50-95): "
        f"{_fmt_number(best_epoch) if best_epoch is not None else 'belirlenemedi'}\n"
        f"HTML rapor: {html_path}\n",
    )
    outputs["html"] = html_path
    outputs["summary"] = summary_path
    return outputs


def _grouped_metric_svg(
    title: str,
    per_class: dict[str, dict[str, Any]],
    destination: Path,
    category_order: Sequence[str] | None = None,
    description: str | None = None,
) -> Path:
    class_order = (
        [name for name in category_order if name in per_class]
        if category_order is not None
        else [name for name in TARGET_NAMES.values() if name in per_class]
    )
    metric_keys = ("p", "r", "ap50", "map50_95")
    width, height = 1200, 760
    left, top, plot_width, plot_height = 90, 165, 1050, 470
    body = [
        f'<text x="60" y="52" class="title">{escape(title)}</text>',
        '<text x="60" y="80" class="subtitle">'
        "Bütün değerler 0–1 aralığındadır ve her barın üzerinde doğrudan gösterilir."
        "</text>",
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height - value * plot_height
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" class="grid"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
                f'class="axis">{value:.1f}</text>',
            ]
        )
    for metric_index, key in enumerate(metric_keys):
        x = 455 + metric_index * 175
        body.extend(
            [
                f'<rect x="{x}" y="112" width="16" height="16" '
                f'fill="{GENERAL_COLORS[metric_index]}"/>',
                f'<text x="{x + 23}" y="125" class="legend">{_friendly_metric(key)}</text>',
            ]
        )
    group_width = plot_width / max(len(class_order), 1)
    bar_width, gap = 48, 11
    for class_index, class_name in enumerate(class_order):
        group_start = left + class_index * group_width
        bars_total = len(metric_keys) * bar_width + (len(metric_keys) - 1) * gap
        first_x = group_start + (group_width - bars_total) / 2
        metrics = per_class[class_name]
        for metric_index, key in enumerate(metric_keys):
            raw = metrics.get(key)
            if raw is None and key == "map50_95":
                raw = metrics.get("ap")
            value = float(raw or 0.0)
            x = first_x + metric_index * (bar_width + gap)
            bar_height = max(0.0, min(1.0, value)) * plot_height
            y = top + plot_height - bar_height
            body.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
                    f'height="{bar_height:.2f}" rx="4" fill="{GENERAL_COLORS[metric_index]}"/>',
                    f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 8, 152):.2f}" '
                    f'text-anchor="middle" class="value">{value:.3f}</text>',
                ]
            )
        body.append(
            f'<text x="{group_start + group_width / 2:.2f}" y="{top + plot_height + 38}" '
            f'text-anchor="middle" font-size="16" font-weight="600">{escape(class_name)}</text>'
        )
    return _save_svg(
        destination,
        title,
        description
        or "Person, bike motorcycle ve car sınıflarının precision, recall, mAP50 ve mAP50-95 değerleri.",
        "\n".join(body),
        width,
        height,
    )


def _speed_svg(speed: dict[str, Any], destination: Path) -> Path:
    ordered = [
        (key, float(value))
        for key, value in speed.items()
        if isinstance(value, (int, float))
    ]
    maximum = max((value for _, value in ordered), default=1.0)
    total = sum(value for _, value in ordered)
    body = [
        '<text x="60" y="52" class="title">Görüntü başına süre dağılımı</text>',
        '<text x="60" y="80" class="subtitle">'
        "Preprocess, inference, loss ve postprocess süreleri; birim milisaniye/görüntü."
        "</text>",
    ]
    for index, (label, value) in enumerate(ordered):
        y = 155 + index * 95
        bar_width = 850 * value / max(maximum, 1e-9)
        body.extend(
            [
                f'<text x="70" y="{y + 28}" font-size="16" font-weight="600">{escape(label)}</text>',
                f'<rect x="230" y="{y}" width="{bar_width:.2f}" height="42" '
                f'rx="5" fill="{GENERAL_COLORS[index % len(GENERAL_COLORS)]}"/>',
                f'<text x="{240 + bar_width:.2f}" y="{y + 28}" class="value">{value:.3f} ms</text>',
            ]
        )
    body.append(
        f'<text x="70" y="{185 + len(ordered) * 95}" font-size="18" font-weight="600">'
        f"Toplam: {total:.3f} ms/görüntü</text>"
    )
    return _save_svg(
        destination,
        "Görüntü başına süre dağılımı",
        "Değerlendirme aşamalarının milisaniye cinsinden süreleri.",
        "\n".join(body),
        1200,
        max(560, 250 + len(ordered) * 95),
    )


def generate_evaluation_report(
    summary_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate class metric and speed charts from evaluation JSON summaries."""

    if not summary_paths:
        raise ValueError("At least one evaluation summary is required")
    summaries = [load_json(path) for path in summary_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    sections: list[str] = []
    for index, summary in enumerate(summaries):
        head = str(summary.get("head", f"run-{index + 1}"))
        metric_path = output_dir / f"class-metrics-{head}.svg"
        outputs[f"metrics_{head}"] = _grouped_metric_svg(
            f"Sınıf metrikleri — {head}",
            summary.get("per_class", {}),
            metric_path,
        )
        sections.append(
            f'<section><img src="{metric_path.name}" alt="{escape(head)} sınıf metrikleri"></section>'
        )
        speed = summary.get("speed_ms_per_image", {})
        if speed:
            speed_path = output_dir / f"speed-{head}.svg"
            outputs[f"speed_{head}"] = _speed_svg(speed, speed_path)
            sections.append(
                f'<section><img src="{speed_path.name}" alt="{escape(head)} hız dağılımı"></section>'
            )
    if len(summaries) > 1:
        head_metrics: dict[str, dict[str, Any]] = {}
        head_order: list[str] = []
        for summary in summaries:
            head = str(summary.get("head", "unknown"))
            head_order.append(head)
            overall = summary.get("overall", {})
            head_metrics[head] = {
                "p": overall.get("metrics/precision(B)", 0.0),
                "r": overall.get("metrics/recall(B)", 0.0),
                "ap50": overall.get("metrics/mAP50(B)", 0.0),
                "map50_95": overall.get("metrics/mAP50-95(B)", 0.0),
            }
        comparison_path = output_dir / "head-comparison.svg"
        outputs["head_comparison"] = _grouped_metric_svg(
            "YOLO26 head karşılaştırması",
            head_metrics,
            comparison_path,
            category_order=head_order,
            description=(
                "End-to-end ve geleneksel NMS başlıklarının genel precision, "
                "recall, mAP50 ve mAP50-95 değerleri."
            ),
        )
        sections.insert(
            0,
            '<section><img src="head-comparison.svg" '
            'alt="YOLO26 head karşılaştırması"></section>',
        )
    table_rows: list[str] = []
    for summary in summaries:
        overall = summary.get("overall", {})
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(summary.get('head', 'unknown')))}</td>"
            f"<td>{float(overall.get('metrics/precision(B)', 0.0)):.4f}</td>"
            f"<td>{float(overall.get('metrics/recall(B)', 0.0)):.4f}</td>"
            f"<td>{float(overall.get('metrics/mAP50(B)', 0.0)):.4f}</td>"
            f"<td>{float(overall.get('metrics/mAP50-95(B)', 0.0)):.4f}</td>"
            f"<td>{float(overall.get('fitness', 0.0)):.4f}</td>"
            "</tr>"
        )
    sections.append(
        "<section><h2>Head karşılaştırması</h2>"
        "<table><thead><tr><th>Head</th><th>Precision</th><th>Recall</th>"
        "<th>mAP50</th><th>mAP50-95</th><th>Fitness</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></section>"
    )
    html_path = output_dir / "evaluation-report.html"
    atomic_write_text(
        html_path,
        _html_page(
            "YOLO26n değerlendirme grafik raporu",
            "Sınıf bazlı kalite, genel metrik ve hız değerlerinin eksiksiz özeti.",
            sections,
        ),
    )
    summary_path = output_dir / "evaluation-report-summary.txt"
    atomic_write_text(
        summary_path,
        "YOLO26N DEĞERLENDİRME GRAFİK RAPORU\n"
        "====================================\n\n"
        f"Kaynak özet sayısı: {len(summary_paths)}\n"
        f"HTML rapor: {html_path}\n"
        + "\n".join(f"Kaynak: {path}" for path in summary_paths)
        + "\n",
    )
    outputs["html"] = html_path
    outputs["summary"] = summary_path
    return outputs


def generate_prediction_report(
    class_counts: dict[str, int],
    image_count: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate an inference class-count chart and a compact HTML report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    maximum = max(class_counts.values(), default=1)
    body = [
        '<text x="60" y="52" class="title">Inference tespit dağılımı</text>',
        f'<text x="60" y="80" class="subtitle">{_fmt_number(image_count)} giriş işlendi; '
        "barlar toplam tespit sayısını gösterir.</text>",
    ]
    for index, class_name in enumerate(TARGET_NAMES.values()):
        value = int(class_counts.get(class_name, 0))
        y = 165 + index * 130
        bar_width = 850 * value / max(maximum, 1)
        body.extend(
            [
                f'<text x="70" y="{y + 34}" font-size="18" font-weight="600">{escape(class_name)}</text>',
                f'<rect x="260" y="{y}" width="{bar_width:.2f}" height="50" '
                f'rx="6" fill="{SERIES_COLORS[class_name]}"/>',
                f'<text x="{275 + bar_width:.2f}" y="{y + 34}" '
                f'class="value">{_fmt_number(value)}</text>',
            ]
        )
    chart_path = _save_svg(
        output_dir / "prediction-class-counts.svg",
        "Inference tespit dağılımı",
        "Üç hedef sınıfın inference boyunca toplam tespit sayıları.",
        "\n".join(body),
        1200,
        620,
    )
    total = sum(class_counts.values())
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{_fmt_number(class_counts.get(name, 0))}</td>"
        f"<td>{_fmt_percent(class_counts.get(name, 0) / max(total, 1))}</td></tr>"
        for name in TARGET_NAMES.values()
    )
    html_path = output_dir / "prediction-report.html"
    atomic_write_text(
        html_path,
        _html_page(
            "Termal inference grafik raporu",
            f"{_fmt_number(image_count)} giriş, {_fmt_number(total)} toplam tespit.",
            [
                '<section><img src="prediction-class-counts.svg" alt="Tespit sınıf dağılımı"></section>',
                "<section><table><thead><tr><th>Sınıf</th><th>Tespit</th><th>Pay</th>"
                "</tr></thead><tbody>"
                + rows
                + "</tbody></table></section>",
            ],
        ),
    )
    return {"chart": chart_path, "html": html_path}
