"""Table "held-out loss does not predict IHEval": next-token loss vs. extraction-compliance retention.

  python ihc/table_heldout_loss.py      # -> results/heldout_loss_table.csv

Loss values come from results/heldout_ce.json (recorded output of
`modal run ihc/quantize_oss_modal.py --ce <checkpoint paths>`, which needs the checkpoints and a GPU;
the curves run of the assistant-only arm is the one exception, see the note in that file).
Retention comes from results/oss_curve.csv (python ihc/plot_oss_curve.py).
C4 loss = blank-line-separated C4 ("c4nl"). The <|endoftext|>-joined value ("c4") is printed for
reference only: gpt-oss degenerates after a mid-sequence <|endoftext|>, so it is not a valid measurement.
"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS

ROWS = [  # label, checkpoint, precision tag in oss_curve.csv
    ("bf16", "gpt-oss-20b-bf16", "bf16"),
    ("RTN", "gpt-oss-20b-rtn-w2-g32", "rtn_w2"),
    ("GPTQ, C4", "gpt-oss-20b-gptq-w2-g32", "gptq_w2"),
    ("GPTQ, chats + system prompt", "gpt-oss-20b-gptq-w2-g32-cal-syschat", "gptq_w2_cal_syschat"),
    ("GPTQ, chats, no system prompt", "gpt-oss-20b-gptq-w2-g32-cal-chat_nosys", "gptq_w2_cal_nosys"),
    ("Block-AP only, C4", "gpt-oss-20b-blockap-w2-g32-cal-c4", "blockap_w2"),
    ("Block-AP only, chats + system prompt", "gpt-oss-20b-blockap-w2-g32-cal-syschat", "blockap_w2_cal_syschat"),
    ("EfficientQAT, C4", "gpt-oss-20b-eqat-w2-g32-cal-c4", "eqat_w2"),
    ("EfficientQAT, chats + system prompt", "gpt-oss-20b-eqat-w2-g32-cal-syschat", "eqat_w2_cal_syschat"),
    ("EfficientQAT, chats, assistant-token loss", "gpt-oss-20b-eqat-w2-g32-cal-syschat-asst", None),
]


def main():
    ce = json.load(open(RESULTS / "heldout_ce.json"))["results"]
    cur = pd.read_csv(RESULTS / "oss_curve.csv")
    ret = cur[(cur.task == "extraction") & cur.kind.str.startswith("compliance")].set_index("precision").retention
    out = []
    for label, ck, prec in ROWS:
        c = ce[f"/ckpt/{ck}"]
        r = 1.0 if prec == "bf16" else ret.get(prec, float("nan")) if prec else float("nan")
        out.append(dict(checkpoint=label, chats_loss=c["syschat"], c4_loss=c["c4nl"],
                        c4_eot_joined_loss_invalid=c["c4"], ext_compliance_retention=r))
    o = pd.DataFrame(out)
    o.to_csv(RESULTS / "heldout_loss_table.csv", index=False)
    print(o.round(3).to_string(index=False))
    print("wrote", RESULTS / "heldout_loss_table.csv")


if __name__ == "__main__":
    main()
