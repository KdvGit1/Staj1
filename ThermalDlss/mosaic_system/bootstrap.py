"""Ana proje modüllerini değiştirmeden yeniden kullanmak için yol yardımcısı."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root() -> Path:
    """`model.py`, `losses.py` ve `dataset.py` importlarını erişilebilir yap."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT

