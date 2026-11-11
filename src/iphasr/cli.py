"""`bench` command line.

    bench report               rebuild results/table.md from cached transcripts (no network); gate
    bench preflight            secret/audio sweep before every push
    bench synth-text N         write the train_synth sentence list (no audio)
    bench record ...           test_human sessions: setup | serve | new | call | fetch
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="rebuild results/table.md from cache; exit 1 if the gate fails")
    p.add_argument("--resamples", type=int, default=10_000)
    sub.add_parser("preflight", help="sweep tracked files for audio, private data and secrets")
    p = sub.add_parser("synth-text", help="write train_synth sentences to the data root")
    p.add_argument("n", type=int)
    p.add_argument("--seed", type=int, default=0)
    from .record_cli import add_parser as add_record
    add_record(sub)
    args = ap.parse_args(argv)

    if args.cmd == "report":
        from .report import main as report_main
        return report_main(args.resamples)
    if args.cmd == "preflight":
        from .preflight import main as pf_main
        return pf_main()
    if args.cmd == "synth-text":
        from .paths import data_root, write_jsonl
        from .synth import generate
        out = data_root() / "synth" / "sentences.jsonl"
        write_jsonl(out, generate(args.n, args.seed))
        print(f"wrote {args.n} sentences to {out}")
        return 0
    if args.cmd == "record":
        from .record_cli import run
        return run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
