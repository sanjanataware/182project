"""Shared paths, I/O helpers, and thin wrappers around IHEval's (unmodified) checkers."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
IHEVAL = ROOT / "third_party" / "IHEval"
BENCH = IHEVAL / "benchmark"
DATA = ROOT / "data"
RESPONSES = ROOT / "responses"
SCORED = ROOT / "scored"
RESULTS = ROOT / "results"

# IHEval source files are used as-is (license: CC BY-NC-ND 4.0, so we do not modify them).
IHEVAL_FILES = {
    ("rule_following", "aligned"): BENCH / "rule-following/single-turn/aligned/default/input_data.json",
    ("rule_following", "conflict"): BENCH / "rule-following/single-turn/conflict/default/input_data.json",
    ("rule_following", "reference"): BENCH / "rule-following/single-turn/reference/default/input_data.json",
    ("extraction", "aligned"): BENCH / "safety/system-prompt-extract/aligned/default/input_data.json",
    ("extraction", "conflict:weak_defense"): BENCH / "safety/system-prompt-extract/conflict/weak_defense/input_data.json",
    ("extraction", "conflict:strong_defense"): BENCH / "safety/system-prompt-extract/conflict/strong_defense/input_data.json",
    ("extraction", "reference"): BENCH / "safety/system-prompt-extract/reference/default/input_data.json",
}

# Nominal bits per precision label, used for numeric regression terms and x-axes.
PRECISION_BITS = {"bf16": 16, "native": 16,  # native = gpt-oss as released (MXFP4 experts, trained for it)
                   "fp16": 16, "w8": 8, "int8": 8, "w4": 4, "int4": 4, "w3": 3, "int3": 3,
                  "gptq_w8": 8, "gptq_w4": 4, "gptq_w3": 3, "sim_w8": 8, "sim_w4": 4, "sim_w3": 3,
                  "rtn_w4": 4, "rtn_w3": 3, "rtn_w2": 2, "rtn_w3_g192": 3, "rtn_w2_g16": 2, "rtn_w2_g8": 2,
                  "gptq_w3_g192": 3, "gptq_w2_g16": 2, "gptq_w2": 2,
                  "gptq_w2_cal_iheval": 2, "gptq_w2_cal_syschat": 2, "gptq_w2_cal_nosys": 2,
                  "eqat_w2": 2, "eqat_w2_cal_syschat": 2,
                  "blockap_w2": 2, "blockap_w2_cal_syschat": 2}


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def messages_sha(messages: list[dict]) -> str:
    """Stable hash of a prompt, used to verify every precision saw byte-identical inputs."""
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- IHEval checkers
_registry = None
_tensortrust = None


def _load_checkers():
    global _registry, _tensortrust
    if _registry is not None:
        return
    if not IHEVAL.exists():
        sys.exit(f"IHEval not found at {IHEVAL}. Run: git clone https://github.com/ytyz1307zzh/IHEval {IHEVAL}")
    # IHEval's checker modules use flat imports (`import instructions`), so put their dirs on sys.path.
    sys.path.insert(0, str(IHEVAL / "src" / "rule_following" / "evaluate"))
    sys.path.insert(0, str(IHEVAL / "src" / "safety" / "evaluate"))
    import instructions_registry  # type: ignore
    from eval_tensortrust import eval_tensortrust  # type: ignore

    _registry = instructions_registry
    _tensortrust = eval_tensortrust


def ifeval_check(response: str, instruction_ids: list[str], kwargs: list[dict], prompt: str) -> list[bool]:
    """Strict IFEval check, same logic as IHEval's evaluation_main.test_instruction_following_strict."""
    _load_checkers()
    out = []
    for iid, kw in zip(instruction_ids, kwargs):
        inst = _registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**(kw or {}))
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        out.append(bool(response.strip() and inst.check_following(response)))
    return out


def ifeval_check_loose(response: str, instruction_ids: list[str], kwargs: list[dict], prompt: str) -> list[bool]:
    """Loose IFEval check (same response variants as IHEval's test_instruction_following_loose)."""
    _load_checkers()
    r = response.split("\n")
    rm_first = "\n".join(r[1:]).strip()
    rm_last = "\n".join(r[:-1]).strip()
    rm_both = "\n".join(r[1:-1]).strip()
    variants = [response, response.replace("*", ""), rm_first, rm_last, rm_both,
                rm_first.replace("*", ""), rm_last.replace("*", ""), rm_both.replace("*", "")]
    out = []
    for iid, kw in zip(instruction_ids, kwargs):
        inst = _registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**(kw or {}))
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        out.append(any(v.strip() and inst.check_following(v) for v in variants))
    return out


def tensortrust_check(response: str, answer: dict) -> bool:
    _load_checkers()
    return bool(_tensortrust(answer, response))
