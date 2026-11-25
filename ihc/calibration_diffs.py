"""Paired differences between GPTQ 2b g32 calibration arms ("Results: calibration data").

  python ihc/calibration_diffs.py      # -> results/calibration_diffs.csv

Difference in pass rate between two calibration arms on identical prompts (per-prompt mean over
seeds 0 and 1, depth 0), with a 95% cluster-bootstrap CI: 2,000 resamples of IHEval example groups,
rng default_rng(0), consumed in the order below.
  chat format     = chats, no system prompt  - C4
  system prompts  = chats + system prompt    - same chats, no system prompt
  in-distribution = IHEval-train             - chats + system prompt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, SCORED

ARMS = ["gptq_w2", "gptq_w2_cal_nosys", "gptq_w2_cal_syschat", "gptq_w2_cal_iheval"]
COMPARISONS = [("system prompts", "gptq_w2_cal_nosys", "gptq_w2_cal_syschat"),
               ("chat format", "gptq_w2", "gptq_w2_cal_nosys"),
               ("in-distribution", "gptq_w2_cal_syschat", "gptq_w2_cal_iheval")]


def main(n_boot: int = 2000):
    rng = np.random.default_rng(0)
    rows = []
    for p in ARMS:
        for s in (0, 1):
            rows += [json.loads(l) for l in open(SCORED / f"gpt-oss-20b__{p}__s{s}.jsonl")]
    d = pd.DataFrame(rows)
    d = d[d.depth == 0]
    d["kind"] = np.where(d.setting == "conflict", "compliance", "ability")
    pi = d.groupby(["precision", "uid", "task", "kind", "group_id"], as_index=False).compliant.mean()
    out = []
    for name, a, b in COMPARISONS:
        for (task, kind), g in pi.groupby(["task", "kind"]):
            x = g[g.precision == a].set_index("uid").compliant
            y = g[g.precision == b].set_index("uid").compliant
            gid = g[g.precision == a].set_index("uid").group_id
            diff = (y - x).dropna()
            grp = gid.loc[diff.index]
            groups = grp.unique()
            by = {k: diff[grp == k].values for k in groups}
            bs = [np.concatenate([by[k] for k in rng.choice(groups, len(groups))]).mean() for _ in range(n_boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(dict(comparison=name, minus=a, arm=b, task=task, kind=kind, diff=diff.mean(),
                            ci_lo=lo, ci_hi=hi, n_prompts=len(diff), n_groups=len(groups)))
    o = pd.DataFrame(out)
    o.to_csv(RESULTS / "calibration_diffs.csv", index=False)
    for _, r in o.iterrows():
        print(f"  {r.comparison:16s} {r.task:15s} {r.kind:10s} {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
    print("wrote", RESULTS / "calibration_diffs.csv")


if __name__ == "__main__":
    main()
