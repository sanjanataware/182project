"""Run generation on Modal GPUs.

  # smoke test: 20 prompts, BF16, seed 0 (cheap; checks the whole path)
  modal run ihc/modal_app.py --precisions bf16 --seeds 0 --limit 20

  # full run: every precision in parallel (one GPU container each), both seeds
  modal run ihc/modal_app.py --precisions bf16,w8,w4,w3

  # fetch results, then score/analyze locally
  mkdir -p responses && modal volume get ihc-responses / responses/ --force
  python ihc/score.py && python ihc/analyze.py

The local ihc/ code and data/ (eval set) are copied into the image, so every container sees
the exact same prompts. Responses go to the Modal volume `ihc-responses`. generate.py is
resumable, so re-running after a timeout or preemption only does the missing prompts.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("IHC_GPU", "A10G")  # 24 GB is plenty for a 1.7B model; e.g. IHC_GPU=L40S for speed

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.2", "transformers==4.56.2", "pyyaml")
    .env({"HF_HOME": "/cache/hf", "VLLM_LOGGING_LEVEL": "WARNING"})
    .add_local_dir(ROOT / "ihc", "/app/ihc", ignore=["tests", "__pycache__"])
    .add_local_dir(ROOT / "data", "/app/data", ignore=["nemotron_pool.jsonl"])
)

app = modal.App("ihc-generate", image=image)
hf_cache = modal.Volume.from_name("ihc-hf-cache", create_if_missing=True)
responses = modal.Volume.from_name("ihc-responses", create_if_missing=True)
ckpts = modal.Volume.from_name("ihc-checkpoints", create_if_missing=True)  # from ihc/quantize_modal.py


VOLS = {"/cache": hf_cache, "/out": responses, "/ckpt": ckpts}


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes=VOLS)
def generate(precision: str, seeds: list[int], limit: int | None = None, config: str = "gpt-oss-20b.yaml",
                     input: str = "eval_test.jsonl"):
    return _generate(precision, seeds, limit, config, input)


@app.function(gpu="L40S", timeout=24 * 60 * 60, volumes=VOLS)
def generate_l40s(precision: str, seeds: list[int], limit: int | None = None, config: str = "gpt-oss-20b.yaml",
                          input: str = "eval_test.jsonl"):
    """Same as generate, on a 48 GB L40S (for 8B models)."""
    return _generate(precision, seeds, limit, config, input)


@app.function(gpu="H100", timeout=24 * 60 * 60, volumes=VOLS)
def generate_h100(precision: str, seeds: list[int], limit: int | None = None, config: str = "gpt-oss-20b.yaml",
                  input: str = "eval_test.jsonl"):
    """Same as generate, on an 80 GB H100 (gpt-oss: vLLM's MXFP4 MoE kernels want Hopper)."""
    return _generate(precision, seeds, limit, config, input)


def _generate(precision, seeds, limit, config, input):
    import subprocess
    import threading

    responses.reload()  # see files from earlier (possibly interrupted) runs, so resume works
    stop = threading.Event()

    def autosave():  # persist finished chunks every 5 min, so preemption/timeouts lose little
        while not stop.wait(300):
            try:
                responses.commit()
            except Exception as e:
                print(f"autosave commit failed: {e}", flush=True)

    threading.Thread(target=autosave, daemon=True).start()
    for s in seeds:
        cmd = ["python", "/app/ihc/generate.py", "--precision", precision, "--seed", str(s),
               "--input", f"/app/data/{input}", "--out-dir", "/out",
               "--config", f"/app/ihc/configs/{config}"]
        if limit:
            cmd += ["--limit", str(limit)]
        print(" ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True, cwd="/app/ihc")
        finally:
            responses.commit()  # persist whatever finished, even on failure
            hf_cache.commit()
    stop.set()
    return precision


@app.function(timeout=24 * 60 * 60)
def orchestrate(jobs: list[tuple[str, int]], limit: int | None = None, config: str = "gpt-oss-20b.yaml",
                gpu: str = "default", input: str = "eval_test.jsonl") -> list[str]:
    """Runs INSIDE Modal and fans out one GPU container per (precision, seed).

    Launching through this single remote call is what makes `modal run --detach` safe: Modal keeps
    the one detached call alive if the local client exits, and the fan-out happens server-side, so
    closing the terminal cannot cancel the GPU jobs (a local starmap would be cancelled)."""
    report = []
    fn = {"L40S": generate_l40s, "H100": generate_h100}.get(gpu, generate)
    for (p, s), res in zip(jobs, fn.starmap([(p, [s], limit, config, input) for p, s in jobs], return_exceptions=True)):
        line = (f"FAILED {config} {p} s{s}: {type(res).__name__}: {str(res)[:300]}" if isinstance(res, BaseException)
                else f"finished {config} {p} s{s}")
        print(line, flush=True)
        report.append(line)
    return report


@app.local_entrypoint()
def main(precisions: str = "bf16", seeds: str = "", limit: int = 0):
    import yaml

    cfg = yaml.safe_load((ROOT / "ihc/configs/gpt-oss-20b.yaml").read_text())
    seed_list = [int(s) for s in seeds.split(",")] if seeds else cfg["decoding"].get("seeds", [0])
    precs = [p.strip() for p in precisions.split(",") if p.strip()]
    unknown = [p for p in precs if p not in cfg["precisions"]]
    if unknown:
        raise SystemExit(f"unknown precisions {unknown}; the config has {list(cfg['precisions'])}")
    jobs = [(p, s) for p in precs for s in seed_list]
    print(f"launching {len(jobs)} GPU jobs on {GPU}: {jobs}" + (f" (limit {limit})" if limit else ""))
    for line in orchestrate.remote(jobs, limit or None):
        print(line)
    print("now run:  mkdir -p responses && modal volume get ihc-responses / responses/ --force")
