"""Generate responses for one precision level and one sampling seed.

  python ihc/generate.py --precision bf16 --seed 0        # vLLM (CUDA GPU)
  python ihc/generate.py --precision bf16 --all-seeds     # every seed listed in the config
  python ihc/generate.py --precision bf16 --seed 0 --backend hf --limit 20   # smoke test, no vLLM

Writes responses/{model_name}__{precision}__s{seed}.jsonl with one row per eval prompt:
  uid, model, precision, seed, prompt_sha,
  reasoning         text inside <think>...</think> (input for the stage-diagnosis judge)
  response          final answer after </think>: the only part IHEval's checkers score
  response_raw      full generation
  think_unfinished  generation hit max_tokens before </think> (response is empty -> fails)
  finish_reason, n_prompt_tokens, n_output_tokens
Resumable: uids already in the output file are skipped.

The prompt string is rendered ONCE with the shared tokenizer's chat template and handed to the
engine as text, so every precision level gets the identical token sequence. Each request's
sampling seed is a hash of (run seed, uid), so runs are reproducible and resumable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from common import DATA, RESPONSES, ROOT, iter_jsonl, read_jsonl

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


DATE_RE = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")


def render(tokenizer, messages, enable_thinking, dec=None):
    dec = dec or {}
    kw = {}
    if dec.get("reasoning_effort"):  # gpt-oss (harmony) template kwarg
        kw["reasoning_effort"] = dec["reasoning_effort"]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=enable_thinking, **kw)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    if dec.get("fixed_date"):
        # gpt-oss's template stamps today's date into the system message; pin it so prompts are
        # identical whenever (and wherever) they are rendered.
        text, n = DATE_RE.subn(f"Current date: {dec['fixed_date']}", text, count=1)
        assert n == 1, "fixed_date set but the template has no 'Current date:' line"
    return text


HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_TAGS = re.compile(r"<\|(start|end|return|call|channel|message|constrain)\|>")


def split_harmony(text: str) -> tuple[str, str, bool]:
    """gpt-oss output: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final
    <|message|>...<|return|>. Needs skip_special_tokens=False. No final channel -> unfinished."""
    if HARMONY_FINAL not in text:
        return HARMONY_TAGS.sub(" ", text).replace("analysis", "", 1).strip(), "", True
    head, ans = text.rsplit(HARMONY_FINAL, 1)
    ans = re.split(r"<\|(?:return|end|call)\|>", ans)[0]
    reasoning = HARMONY_TAGS.sub(" ", head.replace("<|channel|>analysis<|message|>", "")
                                 .replace("<|start|>assistant", "")).strip()
    return reasoning, ans.strip(), False


def split_reasoning(text: str, thinking: bool, fmt: str = "think") -> tuple[str, str, bool]:
    """-> (reasoning, final_answer, think_unfinished).

    Qwen3's template opens <think> in the prompt when thinking is on, so the generation may
    start mid-trace with no opening tag."""
    if fmt == "harmony":
        return split_harmony(text)
    if THINK_CLOSE in text:
        head, tail = text.split(THINK_CLOSE, 1)
        return head.replace(THINK_OPEN, "").strip(), tail.strip(), False
    if thinking or text.lstrip().startswith(THINK_OPEN):
        return text.replace(THINK_OPEN, "").strip(), "", True  # never closed the trace
    return "", text.strip(), False


def sampling_kwargs(dec):
    kw = dict(temperature=dec["temperature"], top_p=dec.get("top_p", 1.0), top_k=dec.get("top_k", -1),
              min_p=dec.get("min_p", 0.0), max_tokens=dec["max_tokens"])
    if "skip_special_tokens" in dec:  # harmony parsing needs the channel tokens kept
        kw["skip_special_tokens"] = dec["skip_special_tokens"]
    return kw


_LLM = None


def prompt_seed(run_seed: int, uid: str) -> int:
    """Per-request seed from (run seed, uid): independent of batch order and of resuming."""
    import hashlib
    return int(hashlib.sha256(f"{run_seed}|{uid}".encode()).hexdigest()[:8], 16)


def run_vllm(spec, dec, prompts, seeds):
    global _LLM
    from vllm import LLM, SamplingParams

    if _LLM is None:
        _LLM = _load_vllm(spec, dec, LLM)
    llm = _LLM
    base = sampling_kwargs(dec)
    # Chunks so results are written (and can be checkpointed) as they finish, not only at the end.
    chunk = 256
    for i in range(0, len(prompts), chunk):
        sps = [SamplingParams(**base, seed=s) for s in seeds[i:i + chunk]]
        for o in llm.generate(prompts[i:i + chunk], sps, use_tqdm=False):
            c = o.outputs[0]
            yield c.text, c.finish_reason, len(o.prompt_token_ids), len(c.token_ids)
        print(f"  {min(i + chunk, len(prompts))}/{len(prompts)} generated", flush=True)


def _load_vllm(spec, dec, LLM):
    kw = dict(model=str(spec["path"]), tokenizer=spec.get("tokenizer"), dtype=spec.get("dtype", "auto"),
              seed=0, max_model_len=spec.get("max_model_len", dec.get("max_model_len", 20480)),
              gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.85))
    if spec.get("quantization"):
        kw["quantization"] = spec["quantization"]
    if spec.get("kv_cache_dtype"):  # optional KV-cache quantization experiment, e.g. "fp8"
        kw["kv_cache_dtype"] = spec["kv_cache_dtype"]
    return LLM(**kw)


def run_hf(spec, dec, prompts, tokenizer, seeds):
    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(spec.get("dtype"), "auto")
    model = AutoModelForCausalLM.from_pretrained(str(spec["path"]), torch_dtype=dtype).to(dev).eval()
    greedy = dec["temperature"] == 0
    for p, s in zip(prompts, seeds):
        torch.manual_seed(s)
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(dev)
        gen_kw = dict(max_new_tokens=dec["max_tokens"], do_sample=not greedy)
        if not greedy:
            gen_kw.update(temperature=dec["temperature"], top_p=dec.get("top_p", 1.0), top_k=dec.get("top_k", 0))
        with torch.no_grad():
            out = model.generate(**ids, **gen_kw)
        new = out[0, ids["input_ids"].shape[1]:]
        stopped = len(new) < dec["max_tokens"] or new[-1].item() == tokenizer.eos_token_id
        yield tokenizer.decode(new, skip_special_tokens=True), ("stop" if stopped else "length"), \
            ids["input_ids"].shape[1], len(new)


def run_identity(spec, dec) -> dict:
    """Everything that must match for responses to belong in the same file."""
    return {"checkpoint": str(spec["path"]), "dtype": spec.get("dtype", "auto"),
            "quantization": spec.get("quantization"), "kv_cache_dtype": spec.get("kv_cache_dtype"),
            "decoding": dec}


def generate_one(cfg, spec, dec, args, seed, rows, tok):
    out_path = args.out_dir / f"{cfg['model_name']}__{args.precision}__s{seed}.jsonl"
    if out_path.exists() and out_path.stat().st_size:
        raw = out_path.read_bytes()
        if not raw.endswith(b"\n"):  # killed mid-write: drop the partial last line before appending
            out_path.write_bytes(raw[: raw.rfind(b"\n") + 1])
            print(f"  dropped a partial trailing line in {out_path.name}")
    existing = list(iter_jsonl(out_path)) if out_path.exists() else []
    # Never mix runs: an existing file must come from the same checkpoint and decoding settings.
    ident = run_identity(spec, dec)
    for old in existing:
        old_ident = {k: old.get(k, v) for k, v in ident.items()}
        if old_ident != ident:
            sys.exit(f"{out_path} was made with {old_ident}, but the config now says {ident}. "
                     "Move or delete that file (or use a new precision name) before generating.")
    done = {r["uid"] for r in existing}
    todo = [r for r in rows if r["uid"] not in done]
    print(f"[seed {seed}] {len(rows)} prompts, {len(done)} already done, {len(todo)} to generate -> {out_path}")
    if not todo:
        return
    thinking = bool(dec.get("enable_thinking", False))
    prompts = [render(tok, r["messages"], thinking, dec) for r in todo]
    fmt = dec.get("format", "think")
    pseeds = [prompt_seed(seed, r["uid"]) for r in todo]
    gen = run_vllm(spec, dec, prompts, pseeds) if args.backend == "vllm" else run_hf(spec, dec, prompts, tok, pseeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for r, (text, fin, n_in, n_out) in zip(todo, gen):
            reasoning, answer, unfinished = split_reasoning(text, thinking, fmt)
            f.write(json.dumps({
                "uid": r["uid"], "model": cfg["model_name"], "precision": args.precision, "seed": seed,
                "prompt_sha": r["prompt_sha"], "reasoning": reasoning, "response": answer,
                "response_raw": text, "think_unfinished": unfinished,
                "finish_reason": fin, "n_prompt_tokens": n_in, "n_output_tokens": n_out,
                "backend": args.backend, **ident,
            }, ensure_ascii=False) + "\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "ihc/configs/gpt-oss-20b.yaml")
    ap.add_argument("--precision", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true", help="run every seed in decoding.seeds")
    ap.add_argument("--input", type=Path, default=DATA / "eval_test.jsonl")
    ap.add_argument("--out-dir", type=Path, default=RESPONSES)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.precision not in cfg["precisions"]:
        sys.exit(f"precision {args.precision!r} not in {list(cfg['precisions'])}")
    spec = dict(cfg["precisions"][args.precision])
    spec.setdefault("tokenizer", cfg["tokenizer"])
    dec = cfg["decoding"]
    seeds = dec.get("seeds", [0]) if args.all_seeds else [args.seed]
    rows = read_jsonl(args.input)[: args.limit]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["tokenizer"])
    for s in seeds:  # the vLLM engine is loaded once and reused across seeds
        generate_one(cfg, spec, dec, args, s, rows, tok)
    print("done")


if __name__ == "__main__":
    main()
