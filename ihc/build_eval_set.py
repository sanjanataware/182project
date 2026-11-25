"""Build the fixed evaluation set: IHEval examples x instruction distance (0/4/8 filler turns).

Output (all deterministic given --seed):
  data/nemotron_pool.jsonl   cached filler exchanges (user, assistant) streamed from Nemotron
  data/eval_test.jsonl       the evaluation prompts (one row per task/setting/example/depth)
  data/eval_train.jsonl      same construction on train-split examples (for LoRA later)
  data/eval_val.jsonl        same, validation split
  data/split_manifest.json   which IHEval ids landed in which split, plus build settings

Design notes
  * Splits are by IHEval example id (extraction: by base id, the part before "-k"), so the
    test set never shares a system prompt / access code with train or val.
  * Rule-following aligned and conflict share ids, so test uses the SAME 150 ids for both
    (paired comparison). Extraction aligned has 3 items per base id (1 correct code,
    2 wrong guesses), so 50 base ids -> 150 items with the natural 1:2 ratio.
  * A "filler turn" is one user+assistant exchange taken verbatim from Nemotron (fixed text,
    never model-generated), so every precision level sees byte-identical prompts.
  * Depths are nested: depth 4 uses the first 4 of the example's 8 fillers.
  * Fillers are chosen per example to NOT demonstrate a violation of that example's system
    rule (rule-following: the filler assistant reply should pass the system's IFEval
    checks; extraction: no "access granted"/access code). When no fully clean set exists we
    take the least-violating fillers and record `filler_violations` so analysis can check it.
  * Filler pool is partitioned by Nemotron conversation: test fillers and train/val fillers
    come from disjoint conversations.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import logging

logging.getLogger("absl").setLevel(logging.CRITICAL)  # langdetect warnings from IFEval checkers

from common import DATA, IHEVAL_FILES, ifeval_check, messages_sha, read_json, read_jsonl, write_jsonl

NEMOTRON_URL = ("https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1"
                "/resolve/main/data/chat_if.jsonl")


# ---------------------------------------------------------------- filler pool
def build_pool(path: Path, n_convs: int, max_words: int, min_words: int) -> list[dict]:
    if path.exists():
        return read_jsonl(path)
    print(f"Streaming first {n_convs} conversations from Nemotron (file is ~7GB; we read only the head)...")
    pool, seen = [], set()
    req = urllib.request.Request(NEMOTRON_URL, headers={"User-Agent": "ihc-research-script"})
    with urllib.request.urlopen(req) as resp:
        for i, raw in enumerate(resp):
            if i >= n_convs:
                break
            conv = json.loads(raw)
            msgs = [m for m in conv["messages"] if m["role"] in ("user", "assistant")]
            for t in range(len(msgs) - 1):
                u, a = msgs[t], msgs[t + 1]
                if u["role"] != "user" or a["role"] != "assistant":
                    continue
                uc, ac = (u["content"] or "").strip(), (a["content"] or "").strip()
                if "<think>" in ac or not uc or not ac:
                    continue
                nw = len(uc.split()) + len(ac.split())
                if not (min_words <= nw <= max_words):
                    continue
                key = (uc[:200], ac[:200])
                if key in seen:
                    continue
                seen.add(key)
                pool.append({"filler_id": f"{conv['uuid']}#{t}", "conv": conv["uuid"], "user": uc,
                             "assistant": ac, "n_words": nw})
    write_jsonl(path, pool)
    print(f"  cached {len(pool)} exchanges -> {path}")
    return pool


def partition_pool(pool: list[dict], seed: int, test_frac: float = 0.5):
    convs = sorted({p["conv"] for p in pool})
    random.Random(seed).shuffle(convs)
    test_convs = set(convs[: int(len(convs) * test_frac)])
    return [p for p in pool if p["conv"] in test_convs], [p for p in pool if p["conv"] not in test_convs]


# ---------------------------------------------------------------- filler selection
def rf_violations(filler: dict, answer: dict, cache: dict) -> int:
    """How many of the system's constraints the filler assistant reply violates.

    Cached per (filler, single constraint), since many examples share constraints."""
    n = 0
    for iid, kw in zip(answer["instruction_id_list"], answer["kwargs"]):
        key = (filler["filler_id"], iid, json.dumps(kw, sort_keys=True))
        if key not in cache:
            cache[key] = not ifeval_check(filler["assistant"], [iid], [kw], prompt=filler["user"])[0]
        n += cache[key]
    return n


def ex_violations(filler: dict, answer: dict) -> int:
    text = (filler["user"] + "\n" + filler["assistant"]).lower()
    code = answer["access_code"].lower()
    bad = 0
    bad += "access granted" in text
    bad += bool(code) and code in text
    bad += bool(re.search(r"\b(access code|password|secret code)\b", text))
    return bad


def choose_fillers(pool, k, rng, violation_fn, max_scan):
    """Pick k fillers minimising violations; random among ties (seeded).

    Scans a seeded random order of the pool, stopping once k clean fillers are found or
    after max_scan candidates (rare rules such as "no commas" may never reach k clean)."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    scored, n_clean = [], 0
    for idx in order[:max_scan]:
        v = violation_fn(pool[idx])
        scored.append((v, idx))
        n_clean += v == 0
        if n_clean >= k:
            break
    scored.sort(key=lambda s: s[0])  # stable: keeps random order within a violation level
    picked = scored[:k]
    return [pool[i] for _, i in picked], [v for v, _ in picked]


# ---------------------------------------------------------------- splits
def base_id(x) -> str:
    return str(x).rsplit("-", 1)[0]


def split_ids(ids, n_test, val_frac, rng, must_test=()):
    ids = sorted(set(ids) - set(must_test))
    rng.shuffle(ids)
    test = list(must_test) + ids[: max(0, n_test - len(must_test))]
    rest = ids[max(0, n_test - len(must_test)):]
    n_val = int(round(len(rest) * val_frac))
    return set(test), set(rest[n_val:]), set(rest[:n_val])


def iheval_commit() -> str | None:
    import subprocess
    from common import IHEVAL
    try:
        return subprocess.check_output(["git", "-C", str(IHEVAL), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=150, help="test examples per task x setting")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--reference-depths", type=int, nargs="*", default=[0],
                    help="depths for the no-system-prompt reference control (general ability)")
    ap.add_argument("--extract-defense", choices=["weak_defense", "strong_defense"], default="weak_defense")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of non-test ids used for validation")
    ap.add_argument("--nemotron-convs", type=int, default=4000, help="conversations streamed for the filler pool")
    ap.add_argument("--filler-max-words", type=int, default=350, help="max words per filler exchange (user+assistant)")
    ap.add_argument("--filler-min-words", type=int, default=20)
    ap.add_argument("--filler-max-scan", type=int, default=800, help="max candidates checked per example")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    max_depth = max(args.depths + (args.reference_depths or [0]))

    src = {k: read_json(v) for k, v in IHEVAL_FILES.items()}
    conflict_key = ("extraction", f"conflict:{args.extract_defense}")

    # ---- splits
    rf_ids = [x["id"] for x in src[("rule_following", "aligned")]]
    rf_test, rf_train, rf_val = split_ids(rf_ids, args.n_test, args.val_frac, rng)

    ex_al_by_base = defaultdict(list)
    for x in src[("extraction", "aligned")]:
        ex_al_by_base[base_id(x["id"])].append(x)
    per_base = len(src[("extraction", "aligned")]) // len(ex_al_by_base)  # 3
    ex_cf_bases = {base_id(x["id"]) for x in src[conflict_key]}
    overlap = sorted(set(ex_al_by_base) & ex_cf_bases)
    rng.shuffle(overlap)
    n_al_bases = args.n_test // per_base
    # Aligned test bases come from bases that also have a conflict example, so those are paired.
    al_test, al_train, al_val = split_ids(ex_al_by_base, n_al_bases, args.val_frac, rng, must_test=overlap[:n_al_bases])
    cf_test, cf_train, cf_val = split_ids(ex_cf_bases - al_train - al_val, args.n_test, args.val_frac, rng,
                                          must_test=[b for b in al_test if b in ex_cf_bases])
    # Any conflict base that is an aligned train/val base goes to the same split (no leakage).
    cf_train |= (ex_cf_bases & al_train)
    cf_val |= (ex_cf_bases & al_val)

    split_of = {}
    for name, (rf, al, cf) in {"test": (rf_test, al_test, cf_test), "train": (rf_train, al_train, cf_train),
                               "val": (rf_val, al_val, cf_val)}.items():
        for i in rf:
            split_of[("rule_following", i)] = name
        for b in al | cf:
            assert split_of.get(("extraction", b), name) == name, f"leak: base {b}"
            split_of[("extraction", b)] = name

    # ---- filler pool
    pool = build_pool(args.out / "nemotron_pool.jsonl", args.nemotron_convs, args.filler_max_words, args.filler_min_words)
    pool_test, pool_trainval = partition_pool(pool, args.seed)
    print(f"filler pool: {len(pool_test)} test / {len(pool_trainval)} train+val exchanges")

    # ---- fillers per example (shared across settings and depths -> paired, nested)
    rf_cache: dict = {}
    fillers_for: dict = {}

    def get_fillers(task, key, answer, split):
        if (task, key) in fillers_for:
            return fillers_for[(task, key)]
        p = pool_test if split == "test" else pool_trainval
        frng = random.Random(f"{args.seed}|{task}|{key}")
        if task == "rule_following":
            fn = lambda f: rf_violations(f, answer, rf_cache)
        else:
            fn = lambda f: ex_violations(f, answer)
        fillers_for[(task, key)] = choose_fillers(p, max_depth, frng, fn, args.filler_max_scan)
        return fillers_for[(task, key)]

    # ---- assemble rows
    rows = defaultdict(list)
    settings = [("rule_following", "aligned", ("rule_following", "aligned")),
                ("rule_following", "conflict", ("rule_following", "conflict")),
                ("rule_following", "reference", ("rule_following", "reference")),
                ("extraction", "aligned", ("extraction", "aligned")),
                ("extraction", "conflict", conflict_key),
                ("extraction", "reference", ("extraction", "reference"))]
    al_answers = {x["id"]: x["answer"] for x in src[("extraction", "aligned")]}

    for task, setting, key in settings:
        depths = args.reference_depths if setting == "reference" else args.depths
        if not depths:
            continue
        for ex in src[key]:
            group = ex["id"] if task == "rule_following" else base_id(ex["id"])
            split = split_of.get((task, group))
            if split is None:
                continue
            # Reference extraction rows have system_prompt=None; fillers are chosen from the aligned answer.
            ans_for_filler = ex["answer"] if not (task == "extraction" and setting == "reference") else al_answers.get(ex["id"], ex["answer"])
            # Rule-following: fillers keyed by id (aligned/conflict/reference share them).
            # Extraction: keyed by base id (all items about the same access code share them).
            fill, viol = get_fillers(task, group, ans_for_filler, split)
            for d in depths:
                msgs = []
                if ex.get("system"):
                    msgs.append({"role": "system", "content": ex["system"]})
                for f in fill[:d]:
                    msgs += [{"role": "user", "content": f["user"]}, {"role": "assistant", "content": f["assistant"]}]
                msgs.append({"role": "user", "content": ex["instruction"]})
                rows[split].append({
                    "uid": f"{task}|{setting}|{ex['id']}|d{d}",
                    "task": task, "setting": setting, "example_id": str(ex["id"]), "group_id": str(group),
                    "depth": d, "split": split,
                    "messages": msgs, "prompt_sha": messages_sha(msgs),
                    "final_instruction": ex["instruction"],
                    "answer": ex["answer"],
                    "filler_ids": [f["filler_id"] for f in fill[:d]],
                    "filler_violations": sum(viol[:d]),
                    "filler_words": sum(f["n_words"] for f in fill[:d]),
                })

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        rs.sort(key=lambda r: r["uid"])
        n = write_jsonl(args.out / f"eval_{split}.jsonl", rs)
        print(f"{split}: {n} prompts -> {args.out / f'eval_{split}.jsonl'}")

    # ---- report
    from collections import Counter
    c = Counter((r["task"], r["setting"], r["depth"]) for r in rows["test"])
    print("\ntest prompts per (task, setting, depth):")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    dirty = [r for r in rows["test"] if r["filler_violations"] > 0]
    print(f"\ntest rows whose fillers violate the system rule: {len(dirty)}/{len(rows['test'])}"
          f"  (recorded as `filler_violations`)")

    manifest = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "iheval_commit": iheval_commit(),
        "rule_following": {"test": sorted(rf_test), "train": sorted(rf_train), "val": sorted(rf_val)},
        "extraction_base_ids": {"test": sorted(al_test | cf_test), "train": sorted(al_train | cf_train),
                                "val": sorted(al_val | cf_val)},
        "counts": {s: len(r) for s, r in rows.items()},
    }
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
