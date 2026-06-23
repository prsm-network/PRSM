#!/usr/bin/env python3
"""Activation-DP privacy/quality sweep (sp1233 Phase 2).

Faithfully measures what the PRODUCTION DP injector does to a real model's
output: a forward hook on a mid-decoder layer injects ``ActivationDPInjector``
noise into that layer's hidden state (== the inter-stage activation a 2-stage
parallax chain would noise), then the forward continues to logits and we decode.
Sweeps ε × clip_mode × clip_norm in ONE model load (no node relaunch).

GPU result (2026-06-23, Qwen2.5-1.5B-Instruct, layer-14 split, δ=1e-5; clean ref
"...Paris."), confirming the ε²/D curse and that the SHIPPED default is broken:
  measured activation norm at split: tensor L2≈11054, per-token L2≈382
  - tensor + clip=1.0 (PRODUCTION DEFAULT): GARBAGE at every ε (8…32768) —
    clip=1.0 shrinks the L2≈11054 signal ~11000× before any noise.
  - token  + clip≈382 (matched per-token): recovers EXACT clean output only at
    ε≈2048/stage (256× the STANDARD tier's ~4/stage) → effectively no privacy.
Takeaway: strong activation-DP and usable quality are mutually exclusive on
high-dim activations; use a FIXED (data-independent) clip hyperparameter for a
valid DP guarantee, and treat DP as weak/best-effort (lean on TEE for real
privacy). Re-run this after any DP mechanism change to re-validate.

Run on a GPU box:  PYTHONPATH=~/PRSM python3 scripts/dp_quality_sweep.py
"""
import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The SAME DP primitive production uses. Prefer the normal import; on a
# minimal-install box (no full `pip install -e .`) fall back to loading the
# module file directly, bypassing prsm.compute.inference.__init__'s heavy chain.
try:
    from prsm.compute.inference.activation_dp import (
        ActivationDPInjector, StageNoisePolicy,
    )
except Exception:  # noqa: BLE001
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _root = _os.environ.get("PRSM_ROOT", _os.path.expanduser("~/PRSM"))
    _ad_path = _os.path.join(_root, "prsm/compute/inference/activation_dp.py")
    _spec = _ilu.spec_from_file_location("_prsm_activation_dp", _ad_path)
    _ad = _ilu.module_from_spec(_spec)
    _sys.modules["_prsm_activation_dp"] = _ad  # dataclass __module__ lookup needs this
    _spec.loader.exec_module(_ad)
    ActivationDPInjector = _ad.ActivationDPInjector
    StageNoisePolicy = _ad.StageNoisePolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--split-layer", type=int, default=-1,
                    help="decoder layer index whose output is noised (default: middle)")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--epsilons", default="8,32,128,512,2048,8192,32768")
    ap.add_argument("--delta", type=float, default=1e-5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if dev == "cuda" else torch.float32,
    ).to(dev).eval()

    layers = model.model.layers
    n_layers = len(layers)
    split = args.split_layer if args.split_layer >= 0 else n_layers // 2
    print(f"[info] {n_layers} layers; injecting at layer {split} output", flush=True)

    # chat template (instruct) — mirror sp1230/1208
    msgs = [{"role": "user", "content": args.prompt}]
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

    def greedy(injector=None):
        cur = ids
        handle = None
        if injector is not None:
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                injector._seen_stages.clear()  # allow re-inject across greedy steps
                noised = injector.inject_stage(hs.float().cpu().numpy(), 0)
                t = torch.tensor(noised, dtype=hs.dtype, device=hs.device)
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            handle = layers[split].register_forward_hook(hook)
        try:
            for _ in range(args.max_new):
                with torch.no_grad():
                    logits = model(cur).logits[:, -1, :]
                nxt = int(logits.argmax(-1))
                cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
        finally:
            if handle is not None:
                handle.remove()
        return tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)

    clean = greedy(None)
    print(f"\n[CLEAN, no DP] -> {clean!r}\n", flush=True)

    # measure activation norms at the split (to pick matched clip values)
    cap = {}
    def cap_hook(m, i, o):
        hs = (o[0] if isinstance(o, tuple) else o).float()
        cap["tensor_norm"] = float(torch.linalg.norm(hs))
        cap["token_norm"] = float(torch.linalg.norm(hs, dim=-1).mean())
    h = layers[split].register_forward_hook(cap_hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    print(f"[norms] tensor L2={cap['tensor_norm']:.1f}  mean per-token L2={cap['token_norm']:.1f}\n", flush=True)

    eps_list = [float(x) for x in args.epsilons.split(",")]
    combos = [
        ("tensor", 1.0), ("tensor", cap["tensor_norm"]),
        ("token", 1.0), ("token", cap["token_norm"]),
    ]
    print(f"{'mode':>7} {'clip':>9} {'eps':>8}  output", flush=True)
    print("-" * 60, flush=True)
    for mode, clip in combos:
        for eps in eps_list:
            pol = StageNoisePolicy(
                per_stage_epsilon=eps, total_expected_epsilon=eps,
                stage_count=1, tier="standard", clip_norm=clip,
                delta=args.delta, enabled=True, clip_mode=mode)
            out = greedy(ActivationDPInjector(pol))
            match = "  <== matches clean" if out.strip() == clean.strip() else ""
            print(f"{mode:>7} {clip:>9.1f} {eps:>8.0f}  {out!r}{match}", flush=True)
    print("\n[done] find the ε where output recovers coherence.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
