import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import modal
from quantize_oss_modal import image, hf_cache, ckpts, BASE
app = modal.App("ihc-ce-diag2", image=image.add_local_python_source("quantize_oss_modal"))

@app.function(gpu="H100", memory=160 * 1024, timeout=3600, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def diag(paths: list[str]):
    import random, torch, torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
    order = list(range(len(ds))); random.Random(0).shuffle(order); order = order[::-1]
    eot = tok.convert_tokens_to_ids("<|endoftext|>"); nl = tok("\n\n", add_special_tokens=False).input_ids
    def pack(sep, n=4, T=2048):
        buf, out = [], []
        for i in order:
            buf += tok(ds[i]["text"], add_special_tokens=False).input_ids + sep
            while len(buf) >= T and len(out) < n: out.append(buf[:T]); buf = buf[T:]
            if len(out) >= n: break
        return torch.tensor(out)
    longdocs = []
    for i in order:
        ids = tok(ds[i]["text"], add_special_tokens=False).input_ids
        if len(ids) >= 2048: longdocs.append(ids[:2048])
        if len(longdocs) == 4: break
    sets = {"sep_eot": pack([eot]), "sep_newlines": pack(nl), "sep_none": pack([]), "long_single_docs": torch.tensor(longdocs)}
    res = {}
    for p in paths:
        m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
        r = {}
        for k, X in sets.items():
            ls = []
            for i in range(len(X)):
                ids = X[i:i+1].cuda()
                with torch.no_grad(): lg = m(input_ids=ids, use_cache=False).logits.float()
                ls.append(F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="none"))
            ls = torch.stack(ls)
            r[k] = {"mean": round(float(ls.mean()), 3), "pos<256": round(float(ls[:, :256].mean()), 3),
                    "256-1024": round(float(ls[:, 256:1024].mean()), 3), ">1024": round(float(ls[:, 1024:].mean()), 3)}
        res[p] = r; print(p, r, flush=True)
        del m; torch.cuda.empty_cache()
    return res

@app.local_entrypoint()
def main():
    import json
    print(json.dumps(diag.remote(["/ckpt/gpt-oss-20b-bf16", "/ckpt/gpt-oss-20b-rtn-w2-g32"]), indent=1))
