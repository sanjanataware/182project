"""Launch generation on the DEPLOYED Modal app, fully independent of this machine.

  modal deploy ihc/modal_app.py                                  # once, and after code/data changes
  python ihc/launch.py --gpu H100 --input eval_test_d0.jsonl --precisions bf16 --seeds 0,1
  python ihc/launch.py --gpu H100 --input eval_test_d0.jsonl --precisions bf16,rtn_w2 --seeds 0,1
  python ihc/launch.py --status <call_id>                        # is it done?

Why not `modal run --detach`: if the local client is killed it can still send a cancellation to
the entry call (this happened). A call spawned on a deployed app has no tie to a local process.

Never launch a precision/seed that is already running: two containers appending to the same
response file would corrupt it. generate.py resumes from what is saved, so relaunching after a
run has FINISHED or DIED is safe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import modal
import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", default=None, help="call id printed by a previous launch")
    ap.add_argument("--config", type=Path, default=ROOT / "ihc/configs/gpt-oss-20b.yaml",
                    help="a file in ihc/configs/ (it is baked into the deployed image)")
    ap.add_argument("--gpu", choices=["default", "L40S", "H100"], default="default")
    ap.add_argument("--input", default="eval_test.jsonl",
                    help="eval file in data/, e.g. eval_test_d0.jsonl (depth 0 only)")
    args = ap.parse_args()

    if args.status:
        call = modal.FunctionCall.from_id(args.status)
        try:
            for line in call.get(timeout=0):
                print(line)
        except TimeoutError:
            print("still running")
        return

    cfg = yaml.safe_load(args.config.read_text())
    precs = [p for p in args.precisions.split(",") if p]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["decoding"].get("seeds", [0])
    unknown = [p for p in precs if p not in cfg["precisions"]]
    if not precs or unknown:
        raise SystemExit(f"--precisions must be from {list(cfg['precisions'])} (unknown: {unknown})")
    jobs = [(p, s) for p in precs for s in seeds]
    orchestrate = modal.Function.from_name("ihc-generate", "orchestrate")
    call = orchestrate.spawn(jobs, args.limit or None, args.config.name, args.gpu, args.input)
    print(f"spawned {len(jobs)} jobs ({cfg['model_name']}, gpu {args.gpu}, {args.input}): {jobs}\ncall id: {call.object_id}\n"
          f"check:  python ihc/launch.py --status {call.object_id}")


if __name__ == "__main__":
    main()
