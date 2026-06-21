#!/bin/bash
# Turnkey PRSM GPU-node setup for a fresh Lambda Cloud (Ubuntu + CUDA) box (sprint 1211).
#
# Encodes every fix the 2026-06-21 Lambda A10 bench had to discover by hand, in order:
#   1. remove the distro's malformed python3-flatbuffers (invalid version blocks pip>=24.1)
#   2. upgrade build tooling (setuptools>=64 for PEP 660 editable installs)
#   3. editable-install PRSM with BOTH [ml] (inference) and [blockchain] (eth_account/web3
#      — required for operator_address derivation + requester-payment signing)
#   4. pin numpy<2 (the distro's torch/scipy/sklearn are numpy-1.x-ABI; numpy 2 crashes them)
#   5. upgrade Pillow (>=9.1 for transformers' PIL.Image.Resampling) + jinja2 (>=3.1 for
#      apply_chat_template / instruct chat templates)
#
# Run on the box (after cloning the repo to ~/PRSM):
#   bash ~/PRSM/scripts/setup_gpu_node.sh
# Idempotent: safe to re-run. Verifies CUDA + eth_account at the end.
set -uo pipefail

REPO="${PRSM_REPO_DIR:-$HOME/PRSM}"
echo "==> PRSM GPU-node setup (repo: $REPO)"
cd "$REPO" || { echo "ERROR: $REPO not found — clone the repo first"; exit 1; }

echo "==> [1/5] removing distro python3-flatbuffers (malformed version blocks pip)"
sudo apt-get remove -y python3-flatbuffers >/dev/null 2>&1 || \
  echo "    (not present / apt unavailable — continuing)"

echo "==> [2/5] upgrading build tooling (setuptools>=64 for PEP 660)"
python3 -m pip install --user --upgrade pip setuptools wheel setuptools-scm >/dev/null

echo "==> [3/5] editable install: .[ml,blockchain] (inference + eth_account/web3)"
python3 -m pip install --user -e '.[ml,blockchain]' --no-build-isolation

echo "==> [4/5] pinning numpy<2 (distro torch/scipy are numpy-1.x ABI)"
python3 -m pip install --user 'numpy<2' >/dev/null

echo "==> [5/5] upgrading Pillow (>=9.1) + jinja2 (>=3.1) for instruct chat templates"
python3 -m pip install --user --upgrade pillow 'jinja2>=3.1.0' >/dev/null

export PATH="$HOME/.local/bin:$PATH"
echo "==> verifying ..."
python3 - <<'PY'
import importlib, sys
ok = True
try:
    import torch
    print(f"   torch {torch.__version__}  cuda={torch.cuda.is_available()}",
          (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(no GPU)"))
    ok = ok and torch.cuda.is_available()
except Exception as e:
    print("   torch: FAIL", e); ok = False
for m in ("transformers", "eth_account", "web3", "jinja2", "PIL"):
    try:
        importlib.import_module(m); print(f"   {m}: ok")
    except Exception as e:
        print(f"   {m}: FAIL {e}"); ok = False
sys.exit(0 if ok else 1)
PY
rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "✅ GPU node ready. 'prsm' at: $(command -v prsm || echo "$HOME/.local/bin/prsm")"
  echo "   (ensure ~/.local/bin is on PATH: export PATH=\$HOME/.local/bin:\$PATH)"
else
  echo "⚠️  setup finished with verification failures above — fix those before starting the node."
fi
exit $rc
