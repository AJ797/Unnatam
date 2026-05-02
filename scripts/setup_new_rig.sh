#!/usr/bin/env bash
# One-shot setup for a fresh rig (e.g. 8× RTX PRO 6000S 96GB).
# Run this from /workspace right after SSH-ing into a clean rig.
#
# Assumptions:
#   - PyTorch image (torch already installed)
#   - venv autoactivates as /venv/main
#   - workspace at /workspace (transferred via vast.ai or fresh clone)
#
# Order:
#   1. Verify GPUs + network
#   2. Install Unnatam package + deps
#   3. Build mamba-ssm and causal-conv1d from source against installed PyTorch
#   4. Verify fast scan kernel works
#   5. Smoke test
#
# After this finishes successfully, run scripts/run_ia_full.sh

set -e

echo "================================================"
echo "  Step 1/5: Network + GPU verification"
echo "================================================"
echo "Public IP:";   curl -s ifconfig.me; echo
echo "HF reachable:"; curl -I https://huggingface.co --max-time 5 | head -1
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "================================================"
echo "  Step 2/5: Clone repo + install package"
echo "================================================"
source /venv/main/bin/activate
cd /workspace
if [ ! -d "Unnatam" ]; then
    echo "Cloning Unnatam..."
    echo "EDIT THIS LINE with your actual repo URL"
    # git clone https://github.com/YOUR_USER/Unnatam.git
    echo "ABORTING — please edit setup_new_rig.sh with your git clone URL"
    exit 1
fi
cd /workspace/Unnatam
pip install -e .
pip install tiktoken datasets bitsandbytes

echo ""
echo "================================================"
echo "  Step 3/5: Rebuild mamba-ssm + causal-conv1d (~10-15 min)"
echo "================================================"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir causal-conv1d
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir mamba-ssm

echo ""
echo "================================================"
echo "  Step 4/5: Verify fast scan kernel"
echo "================================================"
python3 -c "
from unnatam.model.ssm import has_fast_scan
import torch
print('CUDA visible devices:', torch.cuda.device_count())
print('GPU 0:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
assert has_fast_scan(), 'fast_scan FALSE — install issue'
print('fast_scan: True ✓')
"

echo ""
echo "================================================"
echo "  Step 5/5: Synthetic smoke test (single GPU, ~30 sec)"
echo "================================================"
python3 scripts/train.py --size tiny --data synthetic \
    --total_steps 10 --device cuda:0 --no_grad_ckpt --log_interval 2

echo ""
echo "================================================"
echo "  SETUP DONE"
echo "================================================"
echo ""
echo "If you transferred /workspace/data and /workspace/runs from the previous"
echo "rig, you're ready to launch IA training now:"
echo ""
echo "  bash scripts/run_ia_full.sh"
echo ""
echo "Otherwise re-tokenize first:"
echo ""
echo "  nohup python3 scripts/prepare_data.py --out /workspace/data/fineweb \\"
echo "      --num_tokens 10_000_000_000 --val_tokens 50_000_000 --n_workers 16 \\"
echo "      > /workspace/tok.log 2>&1 &"
echo "  disown"
