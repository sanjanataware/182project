"""Diagnose the implausible bf16 C4 loss (10.9 nats)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import modal
from quantize_oss_modal import image, hf_cache, ckpts, packed_chunks, BASE

app = modal.App("ihc-ce-diag", image=image.add_local_python_source("quantize_oss_modal"))

@app.function(gpu="H100", memory=160 * 1024, timeout=3600, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def diag(paths: list[str]):
    import torch, torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    out = {"eot_id": tok.convert_tokens_to_ids("<|endoftext|>"), "sot_id": tok.convert_tokens_to_ids("<|startoftext|>"),
           "vocab": len(tok)}
    hold = packed_chunks("c4", tok, 4, 2048, 0, reverse=True)
    chat = packed_chunks("syschat", tok, 4, 2048, 0, reverse=True)
    ds = load_dataset("allenai/c4", data_files="en/c4-validation.00000-of-00008.json.gz", split="train")
    docs = [ds[i]["text"] for i in range(40)][:20]
    def ce(m, ids, per=False):
        ids = ids.cuda()
        with torch.no_grad():
            lg = m(input_ids=ids, use_cache=False).logits.float()
        l = F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="none")
        return l if per else float(l.mean())
    for p in paths:
        r = {}
        for impl in ["default", "eager"]:
            kw = {} if impl == "default" else {"attn_implementation": "eager"}
            m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
            rr = {"attn": m.config._attn_implementation}
            # same as eval_ce (labels=ids)
            with torch.no_grad():
                rr["packed_c4_hf"] = sum(float(m(input_ids=hold[i:i+1].cuda(), labels=hold[i:i+1].cuda(), use_cache=False).loss) for i in range(4)) / 4
                rr["packed_chat_hf"] = sum(float(m(input_ids=chat[i:i+1].cuda(), labels=chat[i:i+1].cuda(), use_cache=False).loss) for i in range(4)) / 4
            # per-position breakdown on packed c4
            ls = torch.cat([ce(m, hold[i:i+1], per=True) for i in range(4)])
            tgt = hold[:, 1:].reshape(-1).cuda(); prev = hold[:, :-1].reshape(-1).cuda()
            pos = torch.arange(2047).repeat(4).cuda()
            rr["packed_c4_target_eot"] = float(ls[tgt == out["eot_id"]].mean()) if (tgt == out["eot_id"]).any() else None
            rr["packed_c4_after_eot"] = float(ls[prev == out["eot_id"]].mean())
            rr["packed_c4_other"] = float(ls[(tgt != out["eot_id"]) & (prev != out["eot_id"])].mean())
            for a, b in [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2047)]:
                rr[f"packed_c4_pos_{a}_{b}"] = float(ls[(pos >= a) & (pos < b)].mean())
            # single unpacked docs
            for name, pre in [("doc_plain", []), ("doc_sot", [out["sot_id"]])]:
                v = [ce(m, torch.tensor([pre + tok(d, add_special_tokens=False).input_ids[:1024]])) for d in docs]
                rr[name] = sum(v) / len(v)
            # docs wrapped in harmony as assistant final message
            v = []
            for d in docs:
                pre = tok("<|start|>user<|message|>Write a web article.<|end|><|start|>assistant<|channel|>final<|message|>", add_special_tokens=False).input_ids
                body = tok(d, add_special_tokens=False).input_ids[:1024]
                ids = torch.tensor([pre + body]).cuda()
                l = ce(m, ids, per=True)[len(pre) - 1:]
                v.append(float(l.mean()))
            rr["doc_harmony_final"] = sum(v) / len(v)
            r[impl] = rr
            print(p, impl, rr, flush=True)
            del m; torch.cuda.empty_cache()
        out[p] = r
    return out

@app.local_entrypoint()
def main():
    import json
    print(json.dumps(diag.remote(["/ckpt/gpt-oss-20b-bf16", "/ckpt/gpt-oss-20b-rtn-w2-g32"]), indent=1))
