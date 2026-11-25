"""Relaunch the E2E-QP curve-logging runs (results/eqat/_curves/*.json) on the deployed Modal app.

  modal deploy ihc/quantize_oss_modal.py
  python ihc/launch_curves.py            # spawns 3 H100 jobs (~90 min each), prints call ids
  modal volume get ihc-checkpoints /_curves results/eqat/ --force   # when they finish

Requires the Block-AP checkpoints from an earlier build_eqat run on the volume
(/ckpt/_tmp-gpt-oss-20b-eqat-w2-g32-cal-{c4,syschat}/blockap.pt).
"""
import json

import modal

JOBS = {
    "c4": dict(calib="c4", curves=True, save=False, tag_suffix="-curves"),
    "syschat": dict(calib="syschat", curves=True, save=False, tag_suffix="-curves"),
    "syschat_asst": dict(calib="syschat", curves=True, save=True, loss_mask="assistant", tag_suffix="-asst"),
}

if __name__ == "__main__":
    f = modal.Function.from_name("ihc-quantize-oss", "build_eqat")
    print(json.dumps({k: f.spawn(**kw).object_id for k, kw in JOBS.items()}, indent=1))
