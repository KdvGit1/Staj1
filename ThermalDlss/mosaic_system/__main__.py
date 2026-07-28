"""`python -m mosaic_system {inspect,train,evaluate,tune,report}` giriş noktası."""

from __future__ import annotations

import sys


def main() -> int:
    usage = (
        "Kullanım: python -m mosaic_system "
        "{inspect|train|evaluate|tune|report} [seçenekler]"
    )
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage)
        return 0
    command = sys.argv[1]
    remainder = sys.argv[2:]
    if command not in {"inspect", "train", "evaluate", "tune", "report"}:
        print(usage, file=sys.stderr)
        return 2

    if command == "inspect":
        from .inspect_manifest import main as command_main
    elif command == "train":
        from .train import main as command_main
    elif command == "evaluate":
        from .evaluate import main as command_main
    elif command == "tune":
        from .tune import main as command_main
    else:
        from .report import main as command_main
    return command_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
