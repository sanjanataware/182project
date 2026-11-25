"""Simulated low-bit quantization of gpt-oss-20b's MoE expert weights (round-to-nearest).

  modal run ihc/quantize_oss_modal.py                  # bf16 upcast + RTN 4/3/2-bit
  modal run ihc/quantize_oss_modal.py --bits 0,3       # 0 = plain bf16 upcast

gpt-oss ships its expert MLP weights in MXFP4 (4-bit E2M1 values, one power-of-two scale per 32
inputs), trained for that precision; attention, router, embeddings and expert biases are bf16.
Transformers' Mxfp4Config(dequantize=True) upcasts the experts to bf16 EXACTLY, giving the
baseline `bf16` checkpoint (same numbers as the release, served by the same dense kernels as
every quantized level below, so only weight values differ between levels).

Each lower level re-quantizes ONLY the expert weights (gate_up_proj, down_proj), with
round-to-nearest, asymmetric min-max, group size 32 along the input (reduction) dimension -- the
same block size MXFP4 uses -- and stores the dequantized values back as dense bf16.
Why RTN and not GPTQ (used for Qwen): gpt-oss keeps each layer's 32 experts as one fused 3-D
tensor rather than nn.Linear modules, which the GPTQ tooling does not handle reliably, and
calibrating 32 routed experts per layer is a project of its own. RTN is the standard baseline
quantizer, needs no calibration data, and is exactly reproducible.

Checks recorded in ihc_quant_meta.json: <= 2**bits distinct values in every checked group, and
relative weight error vs. the native weights (should grow as bits fall).
Output: /ckpt/gpt-oss-20b-bf16 and /ckpt/gpt-oss-20b-rtn-w{bits}-g32 on volume ihc-checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BASE = "openai/gpt-oss-20b"
NAME = "gpt-oss-20b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "safetensors>=0.6.2",
                 "huggingface_hub>=0.34.4,<1.0", "hf_transfer>=0.1.9", "kernels", "datasets==3.6.0")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # IHEval TRAIN split only (never the test split), for the calibration-source experiment
    .add_local_file(ROOT / "data" / "eval_train.jsonl", "/data/eval_train.jsonl")
)
app = modal.App("ihc-quantize-oss", image=image)
hf_cache = modal.Volume.from_name("ihc-hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("ihc-checkpoints", create_if_missing=True)

EXPERT_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def rtn(w, bits: int, group: int):
    """w: [E, in, out] (x @ w). Asymmetric min-max RTN in groups of `group` along `in`."""
    import torch

    E, I, O = w.shape
    assert I % group == 0, (w.shape, group)
    x = w.float().transpose(1, 2).reshape(E, O, I // group, group)   # groups along input dim
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    maxq = 2 ** bits - 1
    scale = ((hi - lo) / maxq).clamp(min=1e-12)
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    deq = (q - zero) * scale
    return deq.reshape(E, O, I).transpose(1, 2).contiguous().to(w.dtype)


@app.function(gpu="H100", memory=160 * 1024, timeout=4 * 60 * 60, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def build(bits_list: list[int], group: int = 32) -> list[dict]:
    import json
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=Mxfp4Config(dequantize=True),
                                                 dtype=torch.bfloat16, device_map="cuda")
    params = {n: p for n, p in model.named_parameters() if n.endswith(EXPERT_SUFFIXES)}
    assert len(params) == 2 * model.config.num_hidden_layers, sorted(params)[:4]
    print({n: tuple(p.shape) for n, p in list(params.items())[:2]}, flush=True)
    orig = {n: p.detach().to("cpu", copy=True) for n, p in params.items()}  # native values
    tok = AutoTokenizer.from_pretrained(BASE)
    metas = []
    for bits in bits_list:
        out = f"/ckpt/{NAME}-bf16" if bits == 0 else f"/ckpt/{NAME}-rtn-w{bits}-g{group}"
        if os.path.exists(os.path.join(out, "ihc_quant_meta.json")):
            print(f"{out} exists, skipping", flush=True)
            metas.append(json.load(open(os.path.join(out, "ihc_quant_meta.json"))))
            continue
        rel, max_distinct = [], 0
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for n, p in params.items():
                w0 = orig[n].to("cuda")
                w = w0 if bits == 0 else rtn(w0, bits, group)
                p.data.copy_(w)
                rel.append(float((w.float() - w0.float()).norm() / w0.float().norm()))
                if bits:
                    # distinct values per group, on a random sample of (expert, output column) rows
                    E, I, O = w.shape
                    for _ in range(16):
                        e = int(torch.randint(E, (1,), generator=g)); o = int(torch.randint(O, (1,), generator=g))
                        col = w[e, :, o].reshape(-1, group)
                        max_distinct = max(max_distinct, max(len(torch.unique(r)) for r in col))
        if bits:
            assert max_distinct <= 2 ** bits, f"{max_distinct} distinct values in a group (> 2**{bits})"
        model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(out)
        cfg_path = os.path.join(out, "config.json")
        cfg = json.load(open(cfg_path))
        cfg.pop("quantization_config", None)  # dense bf16 now; vLLM must not expect MXFP4 tensors
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        rel.sort()
        meta = dict(base=BASE, bits=bits or "bf16 (exact upcast of native MXFP4)", method="RTN asym min-max" if bits else "none",
                    group_size=group if bits else None, quantized="MoE expert gate_up_proj + down_proj only",
                    tensors=len(rel), weight_rel_err_vs_native=dict(median=rel[len(rel) // 2], max=rel[-1], min=rel[0]),
                    max_distinct_values_per_group=max_distinct if bits else None,
                    versions=dict(torch=str(torch.__version__), transformers=str(__import__("transformers").__version__)))
        json.dump(meta, open(os.path.join(out, "ihc_quant_meta.json"), "w"), indent=1)
        ckpts.commit()
        print(json.dumps(meta), flush=True)
        metas.append(meta)
    hf_cache.commit()
    return metas


# ------------------------------------------------------------------ GPTQ on the fused experts
def calibration_texts(n: int, seed: int, max_chars: int = 8000) -> list[str]:
    """Identical to ihc/quantize_modal.py (Qwen GPTQ): seeded C4 documents, never eval data."""
    import random

    from datasets import load_dataset

    ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
    idx = random.Random(seed).sample(range(len(ds)), n * 3)
    texts = [ds[i]["text"][:max_chars] for i in idx]
    texts = [t for t in texts if len(t.split()) >= 64][:n]
    assert len(texts) == n, len(texts)
    return texts


CALIB_SOURCES = {
    "c4": "allenai/c4 web text, raw (no chat format) -- the default, same as the Qwen GPTQ",
    "iheval": "IHEval TRAIN split prompts (system rule + user turn, incl. depth-4/8 filler history), "
              "rendered with the chat template; no overlap with the test split",
    "syschat": "HuggingFaceTB/smoltalk systemchats-30k: generic multi-turn chats WITH their system prompt",
    "chat_nosys": "the SAME smoltalk conversations with the system message removed (control)",
}


def calibration_ids(source: str, tok, seed: int, target_tokens: int, max_tokens: int = 2048):
    """Token-id tensors for one calibration source, ~target_tokens in total (matched to the C4 run).
    Chat sources are rendered with gpt-oss's own chat template (date pinned, as in generation)."""
    import json
    import random
    import re

    def render(msgs, gen_prompt):
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt)
        return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", "Current date: 2026-01-01", t, count=1)

    if source == "iheval":
        rows = [json.loads(l) for l in open("/data/eval_train.jsonl")]
        random.Random(seed).shuffle(rows)
        texts = (render(r["messages"], True) for r in rows)
    elif source in ("syschat", "chat_nosys"):
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceTB/smoltalk", "systemchats-30k", split="train")
        order = list(range(len(ds)))
        random.Random(seed).shuffle(order)
        def gen():
            for i in order:
                msgs = [{"role": m["role"], "content": m["content"]} for m in ds[i]["messages"]]
                if not msgs or msgs[0]["role"] != "system":
                    continue
                if source == "chat_nosys":
                    msgs = msgs[1:]
                yield render(msgs, False)
        texts = gen()
    else:
        raise ValueError(source)
    ids, total, n_items = [], 0, 0
    for t in texts:
        x = tok(t, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
        ids.append(x.cuda()); total += x.shape[1]; n_items += 1
        if source != "chat_nosys" and total >= target_tokens:
            break
        if source == "chat_nosys" and n_items >= _SYSCHAT_N.get(seed, 10 ** 9):
            break
    return ids, total, n_items


_SYSCHAT_N: dict = {}  # chat_nosys uses exactly as many conversations as syschat did (same rows)


def gptq_batched(W, H, bits: int, group: int, damp: float = 0.01):
    """GPTQ (Frantar et al. 2022) for a batch of experts at once.

    W: [E, rows, cols] (rows = outputs, cols = inputs), H: [E, cols, cols] = sum x x^T over the tokens
    routed to each expert. Same grid as rtn(): asymmetric min-max, `group` columns per (row, group);
    the ONLY difference from RTN is GPTQ's Hessian-weighted error feedback into later columns.
    No act-order. Returns dequantized weights [E, rows, cols] (float32)."""
    import torch

    W = W.float().clone()
    E, R, C = W.shape
    assert C % group == 0
    H = H.float().clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag == 0
    diag[dead] = 1.0
    W.masked_fill_(dead[:, None, :], 0.0)
    diag += damp * diag.mean(-1, keepdim=True)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    maxq = 2 ** bits - 1
    Q = torch.zeros_like(W)
    block = group if group > 128 else 128
    for i1 in range(0, C, block):
        i2 = min(i1 + block, C)
        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:  # group params from the CURRENT (error-updated) weights
                blk = W1[:, :, i:i + group]
                lo, hi = blk.amin(-1), blk.amax(-1)
                scale = ((hi - lo) / maxq).clamp(min=1e-12)
                zero = torch.round(-lo / scale)
            w = W1[:, :, i]
            q = (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero) * scale
            Q1[:, :, i] = q
            err = (w - q) / Hinv1[:, i, i][:, None]
            W1[:, :, i:] -= err[:, :, None] * Hinv1[:, i, i:][:, None, :]
            Err1[:, :, i] = err
        Q[:, :, i1:i2] = Q1
        W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])
    return Q


class _Stop(Exception):
    pass


@app.function(gpu="H100", memory=160 * 1024, timeout=6 * 60 * 60, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def build_gptq(bits: int, group: int = 32, n_calib: int = 256, seed: int = 0, max_tokens: int = 2048,
               dry_layers: int = 0, calib: str = "c4", target_tokens: int = 115724) -> dict:
    """Sequential GPTQ: layer l is calibrated on activations produced by the already-quantized
    layers 0..l-1. For each layer one pass over the calibration set collects, per expert,
      H_gate_up[e] = sum x x^T  over tokens the router sends to e (x = MLP input),
      H_down[e]    = sum a a^T  with a = the expert's SwiGLU activation (original weights of layer l),
    then both fused tensors are quantized with gptq_batched. dry_layers>0: stop after that many
    layers, save nothing, return stats (a cheap correctness check)."""
    import json
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    out = f"/ckpt/{NAME}-gptq-w{bits}-g{group}" + ("" if calib == "c4" else f"-cal-{calib}")
    if not dry_layers and os.path.exists(os.path.join(out, "ihc_quant_meta.json")):
        return json.load(open(os.path.join(out, "ihc_quant_meta.json")))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=Mxfp4Config(dequantize=True),
                                                 dtype=torch.bfloat16, device_map="cuda").eval()
    if calib == "c4":
        texts = calibration_texts(n_calib, seed)
        ids = [tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.cuda() for t in texts]
        n_items = len(ids)
    else:
        if calib == "chat_nosys":  # same conversations as syschat: count them first
            _, _, _SYSCHAT_N[seed] = calibration_ids("syschat", tok, seed, target_tokens, max_tokens)
        ids, _, n_items = calibration_ids(calib, tok, seed, target_tokens, max_tokens)
    n_tok = sum(x.shape[1] for x in ids)
    print(f"calibration {calib}: {n_items} items, {n_tok} tokens", flush=True)
    layers = model.model.layers
    n_layers = dry_layers or len(layers)
    stats = []
    for li in range(n_layers):
        mlp = layers[li].mlp
        ex = mlp.experts
        E, Hd = ex.gate_up_proj.shape[0], ex.gate_up_proj.shape[1]
        alpha, limit = getattr(ex, "alpha", 1.702), getattr(ex, "limit", model.config.swiglu_limit)
        Hg = torch.zeros(E, Hd, Hd, device="cuda")
        Hdn = torch.zeros(E, ex.down_proj.shape[1], ex.down_proj.shape[1], device="cuda")
        cnt = torch.zeros(E, device="cuda")

        def hook(mod, args, kwargs):
            x = (args[0] if args else kwargs["hidden_states"]).reshape(-1, Hd)
            r = mod.router(x)
            idx = r[1]
            assert idx.dim() == 2 and idx.shape[0] == x.shape[0], idx.shape
            xf = x.float()
            for e in torch.unique(idx).tolist():
                sel = (idx == e).any(-1)
                xe = xf[sel]
                Hg[e] += xe.T @ xe
                gu = xe @ ex.gate_up_proj[e].float() + ex.gate_up_proj_bias[e].float()
                gate, up = gu[..., ::2], gu[..., 1::2]
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
                a = (up + 1) * (gate * torch.sigmoid(gate * alpha))
                Hdn[e] += a.T @ a
                cnt[e] += xe.shape[0]
            raise _Stop

        h = mlp.register_forward_pre_hook(hook, with_kwargs=True)
        with torch.no_grad():
            for x in ids:
                try:
                    model(input_ids=x)
                except _Stop:
                    pass
        h.remove()
        empty = (cnt == 0)
        eye = torch.eye(Hd, device="cuda")
        Hg[empty] = eye; Hdn[empty] = torch.eye(Hdn.shape[1], device="cuda")
        with torch.no_grad():
            w0g = ex.gate_up_proj.data.float()             # [E, in, out]
            qg = gptq_batched(w0g.transpose(1, 2), Hg, bits, group).transpose(1, 2)
            w0d = ex.down_proj.data.float()
            qd = gptq_batched(w0d.transpose(1, 2), Hdn, bits, group).transpose(1, 2)
            # proxy for quality: Hessian-weighted output error, GPTQ vs. RTN on the same layer
            def herr(q, w0, H):
                d = (q - w0).transpose(1, 2)              # [E, out, in]
                return float(torch.einsum("eoi,eij,eoj->", d, H, d).clamp(min=0).sqrt())
            rg, rd = rtn(w0g, bits, group).float(), rtn(w0d, bits, group).float()
            st = dict(layer=li, tokens_per_expert_min=int(cnt.min()), empty_experts=int(empty.sum()),
                      gptq_vs_rtn_out_err_gate_up=herr(qg, w0g, Hg) / max(herr(rg, w0g, Hg), 1e-9),
                      gptq_vs_rtn_out_err_down=herr(qd, w0d, Hdn) / max(herr(rd, w0d, Hdn), 1e-9),
                      rel_err_gate_up=float((qg - w0g).norm() / w0g.norm()),
                      rel_err_down=float((qd - w0d).norm() / w0d.norm()))
            # grid check on a sample: <= 2**bits distinct values per (expert, output, group)
            for q in (qg, qd):
                col = q[0, :, 0].reshape(-1, group)
                assert max(len(torch.unique(r)) for r in col) <= 2 ** bits
            ex.gate_up_proj.data.copy_(qg.to(torch.bfloat16))
            ex.down_proj.data.copy_(qd.to(torch.bfloat16))
        stats.append(st)
        print(json.dumps(st), f"{(time.time() - t0) / 60:.1f} min", flush=True)
        del Hg, Hdn
        torch.cuda.empty_cache()
    rel = sorted([s["rel_err_gate_up"] for s in stats] + [s["rel_err_down"] for s in stats])
    ratio = sorted([s["gptq_vs_rtn_out_err_gate_up"] for s in stats] + [s["gptq_vs_rtn_out_err_down"] for s in stats])
    meta = dict(base=BASE, bits=bits, method="GPTQ (own implementation, batched over experts), asym min-max grid",
                group_size=group, act_order=False, damp=0.01, sequential=True,
                quantized="MoE expert gate_up_proj + down_proj only",
                calibration=dict(source=calib, description=CALIB_SOURCES.get(calib, ""), n_items=n_items,
                                 seed=seed, max_tokens=max_tokens, total_tokens=n_tok),
                weight_rel_err_vs_native=dict(median=rel[len(rel) // 2], max=rel[-1], min=rel[0]),
                out_err_gptq_over_rtn=dict(median=ratio[len(ratio) // 2], max=ratio[-1], min=ratio[0]),
                layers_done=n_layers, minutes=round((time.time() - t0) / 60, 1),
                versions=dict(torch=str(torch.__version__), transformers=str(__import__("transformers").__version__)))
    if dry_layers:
        meta["dry_run"] = True
        return meta
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    cfg_path = os.path.join(out, "config.json")
    cfg = json.load(open(cfg_path)); cfg.pop("quantization_config", None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump(dict(meta, per_layer=stats), open(os.path.join(out, "ihc_quant_meta.json"), "w"), indent=1)
    ckpts.commit()
    hf_cache.commit()
    return meta


# ------------------------------------------------------------------ EfficientQAT on the fused experts
# EfficientQAT (Chen et al. 2024): (1) Block-AP: block by block, train ALL weights of the block plus the
# quantizer's step sizes and zero points with a straight-through estimator, to reconstruct the
# full-precision block output; (2) E2E-QP: freeze the integer codes, train only the step sizes end to
# end on next-token loss. The public code targets nn.Linear; here the same algorithm runs on gpt-oss's
# fused [E, in, out] expert tensors. Grid identical to rtn()/GPTQ (asym, 2^bits levels, `group` inputs
# per group) and the init IS rtn(), so RTN, GPTQ and EfficientQAT differ only in how values are chosen.
# Scope choices (only the experts are quantized in this project):
#  * Block-AP trains only the expert weights + their quantizer params; attention, router, norms and
#    biases stay at native values (as in the RTN/GPTQ levels). Because the experts sit after
#    attention inside a block, the post-attention residual is fixed and computed once per layer.
#  * Layer inputs follow the paper: quantized-path inputs, full-precision-path targets.
#  * Budget is smaller than the paper's (4096 x 2048 tokens): n_train x 2048 tokens, same for both
#    phases; paper LRs for 2-bit (Block-AP weight 2e-5 / quant 1e-4, 2 epochs, batch 2;
#    E2E-QP 2e-5, 1 epoch).


def packed_chunks(source: str, tok, n: int, T: int, seed: int, reverse: bool = False, labels: bool = False):
    """n x T token ids packed from one source (C4 docs joined by <|endoftext|>, or rendered smoltalk
    system chats). reverse=True walks the same shuffled order from the end -> disjoint held-out set."""
    import random
    import re

    import torch
    from datasets import load_dataset

    if source in ("c4", "c4nl"):
        # "c4" (training/calibration): docs joined by <|endoftext|>. "c4nl" (held-out CE only): joined by
        # "\n\n" -- gpt-oss mis-predicts badly after a mid-sequence <|endoftext|> (bf16 CE 10.9 vs 4.1 nats).
        ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
        sep = ([tok.convert_tokens_to_ids("<|endoftext|>")] if source == "c4"
               else tok("\n\n", add_special_tokens=False).input_ids)
        get = lambda i: ds[i]["text"]
    elif source == "syschat":
        ds = load_dataset("HuggingFaceTB/smoltalk", "systemchats-30k", split="train")
        sep = []

        def get(i):
            msgs = [{"role": m["role"], "content": m["content"]} for m in ds[i]["messages"]]
            if not msgs or msgs[0]["role"] != "system":
                return None
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", "Current date: 2026-01-01", t, count=1)
    else:
        raise ValueError(source)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    if reverse:
        order = order[::-1]
    # labels=True: also return a mask that keeps only assistant-message tokens (content + its <|end|>/
    # <|return|>), for assistant-only loss; ids and packing are identical to labels=False.
    span = re.compile(r"<\|start\|>assistant[^<]*(?:<\|channel\|>[^<]*)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)", re.S)
    buf, out, mbuf, mout = [], [], [], []
    for i in order:
        t = get(i)
        if t is None:
            continue
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=labels)
        buf += enc.input_ids + sep
        if labels:
            spans = [(m.start(1), m.end()) for m in span.finditer(t)]
            mbuf += [any(a <= st < b for a, b in spans) for st, _ in enc.offset_mapping] + [False] * len(sep)
        while len(buf) >= T and len(out) < n:
            out.append(buf[:T]); buf = buf[T:]
            if labels:
                mout.append(mbuf[:T]); mbuf = mbuf[T:]
        if len(out) >= n:
            break
    assert len(out) == n, (source, len(out))
    if labels:
        ids, m = torch.tensor(out), torch.tensor(mout)
        return ids, torch.where(m, ids, torch.full_like(ids, -100))
    return torch.tensor(out)


def _round_ste(x):
    return (x.round() - x).detach() + x


class _FQ:
    """Learnable fake-quantizer for one fused expert tensor w [E, in, out], groups along `in`."""

    def __init__(self, w, bits: int, group: int):
        import torch

        E, I, O = w.shape
        self.shape, self.maxq = (E, I, O), 2 ** bits - 1
        W = w.float().reshape(E, I // group, group, O)
        lo, hi = W.amin(2, keepdim=True), W.amax(2, keepdim=True)
        s = ((hi - lo) / self.maxq).clamp(min=1e-12)
        self.W = torch.nn.Parameter(W)
        self.s = torch.nn.Parameter(s)
        self.z = torch.nn.Parameter(-lo / s)   # round(z) == rtn()'s zero point

    def deq(self):
        import torch

        s = self.s.clamp(min=1e-8)
        zr = _round_ste(self.z)
        q = torch.clamp(_round_ste(self.W / s) + zr, 0, self.maxq)
        return ((q - zr) * s).reshape(self.shape)

    def codes(self):
        """(q uint8, rounded zero, scale) -- the integer representation E2E-QP starts from."""
        import torch

        with torch.no_grad():
            s = self.s.clamp(min=1e-8)
            zr = self.z.round()
            q = torch.clamp(torch.round(self.W / s) + zr, 0, self.maxq)
            return q.to(torch.uint8), zr.clone(), s.clone()


def _deq_codes(q, zr, s, shape):
    return ((q.float() - zr) * s).reshape(shape)


def _moe(mlp, x, Wgu, Wdn, alpha, limit):
    """gpt-oss MoE MLP with given expert weights (same math as transformers' GptOssExperts). x [N, H]."""
    import torch

    ex = mlp.experts
    r = mlp.router(x)
    scores, idx = r[-2], r[-1]   # 4.56: (dense scores [N, E], idx); newer: (logits, top-k scores, idx)
    out = torch.zeros_like(x)
    for e in torch.unique(idx).tolist():
        ti, ki = torch.where(idx == e)
        gu = x[ti] @ Wgu[e] + ex.gate_up_proj_bias[e]
        gate, up = gu[..., ::2], gu[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        o = ((up + 1) * (gate * torch.sigmoid(gate * alpha))) @ Wdn[e] + ex.down_proj_bias[e]
        out = out.index_add(0, ti, (o * _route_w(scores, ti, ki, e)).to(x.dtype))
    return out


def _route_w(scores, ti, ki, e):
    """routing weight per (token, slot): dense [N, E] scores (transformers 4.56) or top-k [N, k]."""
    dense = scores.shape[-1] == _N_EXPERTS["n"]
    return (scores[ti, e] if dense else scores[ti, ki])[:, None]


_N_EXPERTS: dict = {}


@app.function(gpu="H100", memory=200 * 1024, timeout=20 * 60 * 60, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def build_eqat(bits: int = 2, group: int = 32, calib: str = "c4", n_train: int = 1024, n_hold: int = 16,
               T: int = 2048, seed: int = 0, epochs: int = 2, bs: int = 2, lr_w: float = 2e-5, lr_q: float = 1e-4,
               e2e_lr: float = 2e-5, e2e_accum: int = 4, e2e_epochs: int = 1, dry_layers: int = 0,
               dry_e2e_steps: int = 4, loss_mask: str = "all", curves: bool = False, ce_every: int = 16,
               probe_every: int = 32, n_probe: int = 30, probe_tokens: int = 768, save: bool = True,
               tag_suffix: str = "", curve_dry_steps: int = 0) -> dict:
    """curves=True: E2E-QP only (Block-AP resumed from the saved blockap.pt), logging train loss every step,
    held-out CE every `ce_every` steps and a harmony-format probe (greedy generation on `n_probe` fixed
    IHEval TRAIN prompts) every `probe_every` steps, to /ckpt/_curves/<tag><tag_suffix>.json.
    loss_mask="assistant": E2E loss on assistant-message tokens only (syschat)."""
    import json
    import math
    import os
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    dry = dry_layers > 0
    if dry:
        n_train, n_hold = 16, 4
    tag = f"eqat-w{bits}-g{group}-cal-{calib}"
    out_dir = f"/ckpt/{NAME}-{tag}{tag_suffix}"
    curve_path = f"/ckpt/_curves/{tag}{tag_suffix}.json"
    if curves:
        assert os.path.exists(f"/ckpt/_tmp-{NAME}-{tag}/blockap.pt"), "curves mode resumes a saved Block-AP"
    if not dry and save and os.path.exists(os.path.join(out_dir, "ihc_quant_meta.json")):
        return json.load(open(os.path.join(out_dir, "ihc_quant_meta.json")))
    tmp = f"/ckpt/_tmp-{NAME}-{tag}"
    t0 = time.time()
    log = lambda *a: print(f"[{(time.time() - t0) / 60:6.1f} min]", *a, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=Mxfp4Config(dequantize=True),
                                                 dtype=torch.bfloat16, device_map="cpu").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    L = cfg.num_hidden_layers
    if loss_mask == "assistant":
        assert calib == "syschat", "assistant-only loss needs chat data"
        train_ids, train_lab = packed_chunks(calib, tok, n_train, T, seed, labels=True)
        assert torch.equal(train_ids, packed_chunks(calib, tok, n_train, T, seed))
    else:
        train_ids = packed_chunks(calib, tok, n_train, T, seed); train_lab = train_ids
    hold_ids = {s: packed_chunks(s, tok, n_hold, T, seed, reverse=True) for s in ("c4", "syschat")}
    if curves:
        hold_ids["c4nl"] = packed_chunks("c4nl", tok, n_hold, T, seed, reverse=True)
        _, hold_chat_lab = packed_chunks("syschat", tok, n_hold, T, seed, reverse=True, labels=True)
    log(f"data: {calib} train {tuple(train_ids.shape)}, held-out c4/syschat {n_hold} x {T} each")
    _N_EXPERTS["n"] = cfg.num_local_experts
    assert cfg.num_experts_per_tok != cfg.num_local_experts
    alpha = getattr(model.model.layers[0].mlp.experts, "alpha", 1.702)
    limit = getattr(model.model.layers[0].mlp.experts, "limit", cfg.swiglu_limit)

    def ce_from_hidden(H, ids):
        """mean next-token CE from final-layer hidden states (norm + lm_head on GPU)."""
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(H.shape[0]):
                lg = model.lm_head(model.model.norm(H[i:i + 1])).float()
                tot += float(F.cross_entropy(lg[0, :-1], ids[i, 1:].cuda(), reduction="sum")); n += T - 1
        return tot / n

    stats = dict(block_ap=[], ce={})
    codes = {}
    if os.path.exists(os.path.join(tmp, "blockap.pt")) and not dry:
        codes = torch.load(os.path.join(tmp, "blockap.pt"))
        stats = json.load(open(os.path.join(tmp, "blockap_stats.json")))
        log("resumed Block-AP results from", tmp)
    else:
        # ---- capture per-layer kwargs (masks, rotary) with recorder layers; embed on GPU
        model.model.embed_tokens.cuda(); model.model.rotary_emb.cuda()
        model.model.norm.cuda(); model.lm_head.cuda()
        real = model.model.layers
        rec_kw = {}

        class Rec(torch.nn.Module):
            def __init__(self, i, at):
                super().__init__(); self.i = i; self.attention_type = at

            def forward(self, hidden_states, *a, **kw):
                rec_kw[self.i] = (hidden_states, kw)
                if self.i == L - 1:
                    raise _Stop
                return hidden_states

        model.model.layers = torch.nn.ModuleList(
            [Rec(i, getattr(real[i], "attention_type", None)) for i in range(L)])
        try:
            with torch.no_grad():
                model.model(input_ids=train_ids[:1].cuda(), use_cache=False)
        except _Stop:
            pass
        model.model.layers = real
        layer_kw = {i: {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value")}
                    for i, (_, kw) in rec_kw.items()}
        with torch.no_grad():
            emb = lambda ids: torch.cat([model.model.embed_tokens(ids[i:i + 8].cuda()) for i in range(0, len(ids), 8)])
            x0 = emb(train_ids)
            assert torch.equal(x0[:1], rec_kw[0][0]), "embedding path differs from the model's"
            x_fp, x_q = x0, x0.clone()
            hold = {s: {k: emb(ids) for k in ("fp", "q", "rtn")} for s, ids in hold_ids.items()}
        del rec_kw
        log(f"buffers on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

        def run(layer, X, kw, cb=4):
            with torch.no_grad():
                for i in range(0, X.shape[0], cb):
                    y = layer(X[i:i + cb], **kw)
                    X[i:i + cb] = y[0] if isinstance(y, tuple) else y

        n_lay = dry_layers or L
        for li in range(L):
            layer = model.model.layers[li]
            ex = layer.mlp.experts
            if li >= n_lay:   # dry run: remaining layers just get RTN codes
                codes[li] = {k: _FQ(getattr(ex, k).data.cuda(), bits, group).codes() for k in ("gate_up_proj", "down_proj")}
                codes[li] = {k: tuple(t.cpu() for t in v) for k, v in codes[li].items()}
                continue
            layer.cuda()
            kw = layer_kw[li]
            w0 = {k: getattr(ex, k).data.clone() for k in ("gate_up_proj", "down_proj")}
            if li == 0:   # batched call == single call, and _moe == the HF MoE
                a = layer(x_fp[:2], **kw); a = a[0] if isinstance(a, tuple) else a
                b = layer(x_fp[1:2], **kw); b = b[0] if isinstance(b, tuple) else b
                assert (a[1:2] - b).abs().max() <= 1e-2 * b.abs().max(), "batching changes layer output"
            # (a) full-precision targets
            for s in hold:
                run(layer, hold[s]["fp"], kw)
            hold_q_in = {s: hold[s]["q"].clone() for s in hold}
            # (b) post-attention residual of the quantized path (fixed during training)
            cap = {}

            def pre(mod, args):
                cap["h"] = args[0]
                raise _Stop

            hk = layer.post_attention_layernorm.register_forward_pre_hook(pre)
            fp_out = torch.empty_like(x_fp[:1])
            with torch.no_grad():
                for i in range(0, n_train, 4):
                    xi = x_q[i:i + 4]
                    try:
                        layer(xi, **kw)
                    except _Stop:
                        pass
                    x_q[i:i + 4] = cap["h"]
            hk.remove()
            run(layer, x_fp, kw)
            if li == 0:
                with torch.no_grad():   # our MoE reimplementation reproduces the layer (x_q == x_fp at layer 0)
                    h = x_q[:1].reshape(-1, x_q.shape[-1])
                    y = h + _moe(layer.mlp, layer.post_attention_layernorm(h), w0["gate_up_proj"], w0["down_proj"], alpha, limit)
                    rel = float((y - x_fp[0]).float().norm() / x_fp[0].float().norm())
                    assert rel < 1e-2, f"_moe mismatch {rel}"
                    stats["moe_check_rel_err"] = rel
            # (c) Block-AP
            fq = {k: _FQ(w0[k], bits, group) for k in w0}
            opt = torch.optim.AdamW([{"params": [fq[k].W for k in fq], "lr": lr_w},
                                     {"params": [p for k in fq for p in (fq[k].s, fq[k].z)], "lr": lr_q}],
                                    weight_decay=0.0)
            n_steps = epochs * (n_train // bs)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / n_steps)))
            Hd = x_q.shape[-1]
            norm = layer.post_attention_layernorm

            def step_loss(idx):
                h = x_q[idx].reshape(-1, Hd)
                Wgu, Wdn = fq["gate_up_proj"].deq().to(torch.bfloat16), fq["down_proj"].deq().to(torch.bfloat16)
                pred = h + _moe(layer.mlp, norm(h), Wgu, Wdn, alpha, limit)
                return F.mse_loss(pred.float(), x_fp[idx].reshape(-1, Hd).float())

            def hold_mse(Wd):
                for k in Wd:
                    getattr(ex, k).data.copy_(Wd[k].to(torch.bfloat16))
                s = calib
                X = hold_q_in[s].clone(); run(layer, X, kw)
                return float(((X.float() - hold[s]["fp"].float()) ** 2).mean()), X

            with torch.no_grad():
                init_w = {k: fq[k].deq() for k in fq}
            mse_init, _ = hold_mse(init_w)
            del init_w
            losses, g = [], torch.Generator().manual_seed(seed + li)
            for ep in range(epochs):
                perm = torch.randperm(n_train, generator=g)
                for j in range(n_train // bs):
                    loss = step_loss(perm[j * bs:(j + 1) * bs].cuda())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step(); sched.step()
                    losses.append(float(loss))
                    if dry and len(losses) >= 8:
                        break
            # (d) finalize: integer codes -> dequantized weights in the layer
            c = {k: fq[k].codes() for k in fq}
            final = {k: _deq_codes(*c[k], fq[k].shape) for k in fq}
            mse_final, Xq = hold_mse(final)
            hold[calib]["q"] = Xq
            for s in hold:
                if s != calib:
                    hold[s]["q"] = hold_q_in[s]; run(layer, hold[s]["q"], kw)
            # RTN path for the held-out sets (RTN == the Block-AP init)
            for k in w0:
                getattr(ex, k).data.copy_(rtn(w0[k], bits, group))
            for s in hold:
                run(layer, hold[s]["rtn"], kw)
            for k in final:
                getattr(ex, k).data.copy_(final[k].to(torch.bfloat16))
            # (e) next layer's quantized-path inputs
            with torch.no_grad():
                Wgu, Wdn = ex.gate_up_proj.data, ex.down_proj.data
                for i in range(0, n_train, 4):
                    h = x_q[i:i + 4].reshape(-1, Hd)
                    x_q[i:i + 4] = (h + _moe(layer.mlp, norm(h), Wgu, Wdn, alpha, limit)).reshape(-1, T, Hd)
            k0 = max(1, len(losses) // 16)
            st = dict(layer=li, steps=len(losses), train_loss_first=sum(losses[:k0]) / k0,
                      train_loss_last=sum(losses[-k0:]) / k0, heldout_mse_rtn_init=mse_init,
                      heldout_mse_final=mse_final, heldout_mse_ratio=mse_final / max(mse_init, 1e-12),
                      frac_codes_changed_vs_rtn=float(sum(
                          (c[k][0] != _FQ(w0[k], bits, group).codes()[0]).float().mean() for k in c) / 2),
                      rel_err_vs_native=float(sum((final[k] - w0[k].float()).norm() / w0[k].float().norm()
                                                  for k in final) / 2))
            stats["block_ap"].append(st)
            log(json.dumps(st))
            codes[li] = {k: tuple(t.cpu() for t in c[k]) for k in c}
            del fq, opt, final, c, hold_q_in
            layer.cpu(); torch.cuda.empty_cache()
        if not dry:
            for s in hold:
                stats["ce"][s] = {k: ce_from_hidden(hold[s][k], hold_ids[s]) for k in ("fp", "rtn", "q")}
            stats["ce_labels"] = {"fp": "bf16 (native values)", "rtn": "RTN init", "q": "after Block-AP"}
            log("held-out CE", json.dumps(stats["ce"]))
            os.makedirs(tmp, exist_ok=True)
            torch.save(codes, os.path.join(tmp, "blockap.pt"))
            json.dump(stats, open(os.path.join(tmp, "blockap_stats.json"), "w"), indent=1)
            ckpts.commit()
        del x_fp, x_q, x0, hold
        torch.cuda.empty_cache()

    # ---- E2E-QP: integer codes and zero points frozen, step sizes trained on next-token loss
    shapes, wcache = {}, {}
    for li in range(L):
        ex = model.model.layers[li].mlp.experts
        for k in ("gate_up_proj", "down_proj"):
            q, zr, s = codes[li][k]
            shapes[(li, k)] = tuple(getattr(ex, k).shape)
            del ex._parameters[k]
            ex.register_buffer(f"q_{k}", q); ex.register_buffer(f"z_{k}", zr)
            setattr(ex, f"s_{k}", torch.nn.Parameter(s.float()))

        def fwd(hidden_states, router_indices=None, routing_weights=None, _ex=ex, _li=li):
            if _li in wcache:   # frozen weights during probe generation (no grad): dequantize once
                Wgu, Wdn = wcache[_li]
            else:
                Wgu = _deq_codes(_ex.q_gate_up_proj, _ex.z_gate_up_proj, _ex.s_gate_up_proj, shapes[(_li, "gate_up_proj")])
                Wdn = _deq_codes(_ex.q_down_proj, _ex.z_down_proj, _ex.s_down_proj, shapes[(_li, "down_proj")])
                Wgu, Wdn = Wgu.to(hidden_states.dtype), Wdn.to(hidden_states.dtype)
                if wcache.get("on"):
                    wcache[_li] = (Wgu, Wdn)
            in_shape = hidden_states.shape   # 4.56 passes [B, T, H] and expects it back
            hidden_states = hidden_states.reshape(-1, in_shape[-1])
            out = torch.zeros_like(hidden_states)
            for e in torch.unique(router_indices).tolist():
                ti, ki = torch.where(router_indices == e)
                gu = hidden_states[ti] @ Wgu[e] + _ex.gate_up_proj_bias[e]
                gate, up = gu[..., ::2], gu[..., 1::2]
                gate = gate.clamp(max=limit); up = up.clamp(min=-limit, max=limit)
                o = ((up + 1) * (gate * torch.sigmoid(gate * alpha))) @ Wdn[e] + _ex.down_proj_bias[e]
                out = out.index_add(0, ti, (o * _route_w(routing_weights, ti, ki, e)).to(hidden_states.dtype))
            return out.reshape(in_shape)

        ex.forward = fwd
    del codes
    model.cuda()
    scales = [p for n, p in model.named_parameters() if ".s_" in n]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in scales:
        p.requires_grad_(True)
    log(f"E2E-QP: {sum(p.numel() for p in scales) / 1e6:.0f}M step sizes trainable; "
        f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def ce_model(ids):
        model.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(ids.shape[0]):
                tot += float(model(input_ids=ids[i:i + 1].cuda(), labels=ids[i:i + 1].cuda(), use_cache=False).loss)
        return tot / ids.shape[0]

    def ce_curve():
        """held-out CE on c4nl (blank-line separators), chats (all tokens) and chats (assistant tokens)."""
        model.eval(); r = {"c4nl": 0.0, "syschat": 0.0, "syschat_asst": 0.0}
        with torch.no_grad():
            for i in range(n_hold):
                ids = hold_ids["c4nl"][i:i + 1].cuda()
                r["c4nl"] += float(model(input_ids=ids, labels=ids, use_cache=False).loss) / n_hold
                ids = hold_ids["syschat"][i:i + 1].cuda(); lab = hold_chat_lab[i:i + 1].cuda()
                lg = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                r["syschat"] += float(F.cross_entropy(lg, ids[0, 1:])) / n_hold
                r["syschat_asst"] += float(F.cross_entropy(lg, lab[0, 1:], ignore_index=-100)) / n_hold
                del lg
        return r

    probe_prompts = []
    if curves:
        import random as _r
        import re as _re
        items = [json.loads(l) for l in open("/data/eval_train.jsonl")]
        items = [x for x in items if x["depth"] == 0]
        _r.Random(0).shuffle(items)
        per = {}
        for x in items:   # balanced over task x setting
            k = (x["task"], x["setting"]); per.setdefault(k, [])
            if len(per[k]) < -(-n_probe // 5):
                per[k].append(x)
        pool = [x for k in sorted(per) for x in per[k]][:n_probe]
        for x in pool:
            t = tok.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True,
                                        reasoning_effort="medium")
            probe_prompts.append(_re.sub(r"Current date: \d{4}-\d{2}-\d{2}", "Current date: 2026-01-01", t, count=1))
        stop = [tok.convert_tokens_to_ids(x) for x in ("<|return|>", "<|call|>")]

    def probe(step):
        """greedy generations on the fixed prompts -> how many invent a turn / skip reasoning."""
        model.eval(); was_ckpt = model.is_gradient_checkpointing
        if was_ckpt:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = "<|endoftext|>"
        outs = []
        wcache["on"] = True
        with torch.no_grad():
            for b in range(0, len(probe_prompts), 15):
                enc = tok(probe_prompts[b:b + 15], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                g = model.generate(**enc, max_new_tokens=probe_tokens, do_sample=False, eos_token_id=stop,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
                outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=False)
        wcache.clear(); torch.cuda.empty_cache()
        model.config.use_cache = False
        if was_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        outs = [o.replace(tok.pad_token, "") for o in outs]
        r = dict(step=step, n=len(outs),
                 invents_turn=sum(bool(_re.search(r"<\|start\|>(user|developer|system)", o)) for o in outs),
                 skips_reasoning=sum(not o.startswith("<|channel|>analysis") for o in outs),
                 reaches_final=sum("<|channel|>final" in o for o in outs),
                 examples=outs[:3])
        return r

    curve = dict(config=dict(calib=calib, loss_mask=loss_mask, ce_every=ce_every, probe_every=probe_every,
                             n_probe=n_probe, probe_tokens=probe_tokens, n_hold=n_hold),
                 train=[], heldout=[], probe=[])

    def dump_curve():
        os.makedirs(os.path.dirname(curve_path), exist_ok=True)
        json.dump(curve, open(curve_path, "w"), indent=1); ckpts.commit()

    if curves:
        tp = time.time(); curve["probe"].append(probe(0)); log("probe", json.dumps({k: v for k, v in curve["probe"][-1].items() if k != "examples"}), f"{time.time() - tp:.0f}s")
        tc = time.time(); curve["heldout"].append(dict(step=0, **ce_curve())); log("heldout", json.dumps(curve["heldout"][-1]), f"{time.time() - tc:.0f}s")
    if not dry:  # consistency: this path reproduces the streamed Block-AP CE
        stats["ce_blockap_fullmodel"] = {s: ce_model(ids) for s, ids in hold_ids.items()}
        log("Block-AP CE via full model", json.dumps(stats["ce_blockap_fullmodel"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(scales, lr=e2e_lr, weight_decay=0.0)
    n_opt = e2e_epochs * (n_train // e2e_accum)
    if dry:
        n_opt = dry_e2e_steps
    full_n_opt = n_opt
    if curve_dry_steps:   # timing test: same schedule, stop early, evaluate often
        n_opt, ce_every, probe_every = curve_dry_steps, curve_dry_steps, curve_dry_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / full_n_opt)))
    model.train()
    e2e_losses, g = [], torch.Generator().manual_seed(seed + 1000)
    step = 0
    for ep in range(e2e_epochs):
        perm = torch.randperm(n_train, generator=g)
        for j in range(n_train // e2e_accum):
            if step >= n_opt:
                break
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for i in perm[j * e2e_accum:(j + 1) * e2e_accum].tolist():
                x = train_ids[i:i + 1].cuda(); y = train_lab[i:i + 1].cuda()
                loss = model(input_ids=x, labels=y, use_cache=False).loss / e2e_accum
                loss.backward(); tot += float(loss)
            opt.step(); sched.step(); step += 1
            e2e_losses.append(tot)
            if curves:
                curve["train"].append(dict(step=step, loss=tot, lr=sched.get_last_lr()[0]))
                if step % ce_every == 0:
                    curve["heldout"].append(dict(step=step, **ce_curve())); log("heldout", json.dumps(curve["heldout"][-1]))
                if step % probe_every == 0:
                    curve["probe"].append(probe(step))
                    log("probe", json.dumps({k: v for k, v in curve["probe"][-1].items() if k != "examples"}))
                if step % ce_every == 0 or step % probe_every == 0:
                    model.train(); dump_curve()
            if step % 16 == 0 or dry:
                log(f"E2E step {step}/{n_opt} loss {tot:.4f}")
    model.gradient_checkpointing_disable()
    k0 = max(1, len(e2e_losses) // 8)
    stats["e2e"] = dict(steps=step, loss_first=sum(e2e_losses[:k0]) / k0, loss_last=sum(e2e_losses[-k0:]) / k0)
    stats["ce_final"] = {s: ce_model(ids) for s, ids in hold_ids.items()}
    log("final held-out CE", json.dumps(stats["ce_final"]))
    if curves:
        curve["e2e"] = stats["e2e"]; curve["ce_final"] = stats["ce_final"]
        curve["ce_blockap_fullmodel"] = stats.get("ce_blockap_fullmodel")
        curve["minutes"] = round((time.time() - t0) / 60, 1)
        dump_curve()
        if not save or curve_dry_steps:
            return curve

    # ---- materialize dense bf16 weights on the grid, check, save like every other level
    rel, max_distinct = [], 0
    native = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=Mxfp4Config(dequantize=True),
                                                  dtype=torch.bfloat16, device_map="cpu") if not dry else None
    with torch.no_grad():
        for li in range(L):
            ex = model.model.layers[li].mlp.experts
            for k in ("gate_up_proj", "down_proj"):
                W = _deq_codes(getattr(ex, f"q_{k}"), getattr(ex, f"z_{k}"), getattr(ex, f"s_{k}"), shapes[(li, k)])
                Wb = W.to(torch.bfloat16)
                for e, o in ((0, 0), (5, 17), (31, 1000)):
                    col = Wb[e, :, o].reshape(-1, group)
                    max_distinct = max(max_distinct, max(len(torch.unique(r)) for r in col))
                if native is not None:
                    w0 = getattr(native.model.layers[li].mlp.experts, k).data.float()
                    rel.append(float((Wb.float().cpu() - w0).norm() / w0.norm()))
                for b in ("q_", "z_"):
                    del ex._buffers[b + k]
                delattr(ex, f"s_{k}")
                ex.register_parameter(k, torch.nn.Parameter(Wb, requires_grad=False))
            del ex.forward
    assert max_distinct <= 2 ** bits, max_distinct
    meta = dict(base=BASE, bits=bits, group_size=group,
                method="EfficientQAT (Block-AP + E2E-QP; own implementation on fused experts), asym grid, RTN init",
                quantized="MoE expert gate_up_proj + down_proj only",
                calibration=dict(source=calib, description=CALIB_SOURCES.get(calib, ""), n_seqs=n_train, seq_len=T,
                                 total_tokens=n_train * T, seed=seed),
                hyper=dict(block_ap=dict(epochs=epochs, batch=bs, lr_weight=lr_w, lr_quant=lr_q, sched="cosine"),
                           e2e_qp=dict(epochs=e2e_epochs, batch=e2e_accum, lr=e2e_lr, sched="cosine",
                                       trained="step sizes only", grad_ckpt=True, loss_mask=loss_mask)),
                max_distinct_values_per_group=max_distinct,
                minutes=round((time.time() - t0) / 60, 1),
                versions=dict(torch=str(torch.__version__), transformers=str(__import__("transformers").__version__)),
                **{k: v for k, v in stats.items() if k != "block_ap"})
    if rel:
        rel.sort(); meta["weight_rel_err_vs_native"] = dict(median=rel[len(rel) // 2], max=rel[-1], min=rel[0])
    if dry:
        meta["dry_run"] = True; meta["block_ap"] = stats["block_ap"]
        return meta
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out_dir)
    cfg_path = os.path.join(out_dir, "config.json")
    c = json.load(open(cfg_path)); c.pop("quantization_config", None)
    c["use_cache"] = True
    json.dump(c, open(cfg_path, "w"), indent=2)
    json.dump(dict(meta, per_layer=stats["block_ap"]), open(os.path.join(out_dir, "ihc_quant_meta.json"), "w"), indent=1)
    ckpts.commit(); hf_cache.commit()
    return meta


@app.function(memory=200 * 1024, cpu=8, timeout=2 * 60 * 60, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def save_blockap_only(bits: int = 2, group: int = 32, calib: str = "c4") -> dict:
    """Ablation: EfficientQAT stopped after Block-AP (no E2E-QP), i.e. pure layer-output reconstruction
    like GPTQ, no next-token training. Materializes the saved Block-AP codes as a dense bf16 checkpoint."""
    import json
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    tag = f"eqat-w{bits}-g{group}-cal-{calib}"
    out = f"/ckpt/{NAME}-blockap-w{bits}-g{group}-cal-{calib}"
    if os.path.exists(os.path.join(out, "ihc_quant_meta.json")):
        return json.load(open(os.path.join(out, "ihc_quant_meta.json")))
    tmp = f"/ckpt/_tmp-{NAME}-{tag}"
    codes = torch.load(os.path.join(tmp, "blockap.pt"))
    stats = json.load(open(os.path.join(tmp, "blockap_stats.json")))
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=Mxfp4Config(dequantize=True),
                                                 dtype=torch.bfloat16, device_map="cpu")
    rel, max_distinct = [], 0
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            for k in ("gate_up_proj", "down_proj"):
                p = getattr(ex, k)
                W = _deq_codes(*codes[li][k], tuple(p.shape)).to(torch.bfloat16)
                for e, o in ((0, 0), (5, 17), (31, 1000)):
                    col = W[e, :, o].reshape(-1, group)
                    max_distinct = max(max_distinct, max(len(torch.unique(r)) for r in col))
                rel.append(float((W.float() - p.float()).norm() / p.float().norm()))
                p.data.copy_(W)
    assert max_distinct <= 2 ** bits, max_distinct
    rel.sort()
    meta = dict(base=BASE, bits=bits, group_size=group, method="EfficientQAT Block-AP only (no E2E-QP), RTN init",
                quantized="MoE expert gate_up_proj + down_proj only", calibration=dict(source=calib),
                weight_rel_err_vs_native=dict(median=rel[len(rel) // 2], max=rel[-1], min=rel[0]),
                max_distinct_values_per_group=max_distinct, ce=stats.get("ce"),
                versions=dict(torch=str(torch.__version__), transformers=str(__import__("transformers").__version__)))
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    cfg_path = os.path.join(out, "config.json")
    c = json.load(open(cfg_path)); c.pop("quantization_config", None)
    json.dump(c, open(cfg_path, "w"), indent=2)
    json.dump(dict(meta, per_layer=stats.get("block_ap")), open(os.path.join(out, "ihc_quant_meta.json"), "w"), indent=1)
    ckpts.commit(); hf_cache.commit()
    return meta



@app.function(gpu="H100", memory=160 * 1024, timeout=3 * 60 * 60, volumes={"/cache": hf_cache, "/ckpt": ckpts})
def eval_ce(paths: list[str], n_hold: int = 16, T: int = 2048, seed: int = 0,
            sources: tuple = ("c4", "c4nl", "syschat")) -> dict:
    """Held-out next-token CE (same C4 / system-chat held-out chunks as build_eqat) for saved checkpoints."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    hold = {s: packed_chunks(s, tok, n_hold, T, seed, reverse=True) for s in sources}
    res = {}
    for p in paths:
        m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
        with torch.no_grad():
            res[p] = {s: sum(float(m(input_ids=ids[i:i + 1].cuda(), labels=ids[i:i + 1].cuda(), use_cache=False).loss)
                             for i in range(n_hold)) / n_hold for s, ids in hold.items()}
        print(p, res[p], flush=True)
        del m; torch.cuda.empty_cache()
    return res


@app.local_entrypoint()
def main(bits: str = "0,4,3,2", group: int = 32, gptq: bool = False, dry_layers: int = 0, n_calib: int = 256,
         calib: str = "c4", eqat: bool = False, ce: str = ""):
    """RTN by default. --gptq: GPTQ, one container per bit width (bits like "3,2", group shared).
    --eqat: EfficientQAT, one container per calibration source (--calib c4,syschat; --dry-layers N to test).
    --ce path1,path2: held-out CE of saved checkpoints."""
    import json

    if ce:
        print(json.dumps(eval_ce.remote(ce.split(",")), indent=1))
        return
    if eqat:
        jobs = [dict(bits=int(b), group=group, calib=c, dry_layers=dry_layers) for b in bits.split(",") for c in calib.split(",")]
        for m in build_eqat.map([j["bits"] for j in jobs], [j["group"] for j in jobs], [j["calib"] for j in jobs],
                                kwargs={"dry_layers": dry_layers}):
            print(json.dumps(m, indent=1))
        return

    if gptq:
        jobs = [(int(b), group, n_calib, 0, 2048, dry_layers, c) for b in bits.split(",") for c in calib.split(",")]
        for m in build_gptq.starmap(jobs):
            print(json.dumps(m))
        return
    for m in build.remote([int(b) for b in bits.split(",")], group):
        print(json.dumps(m))
