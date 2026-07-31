#!/bin/bash
# ajnano-voice Linux/DGX Setup
# Usage: bash scripts/setup.sh
set -e

echo "=== ajnano-voice Linux Setup ==="

ARCH=$(uname -m)
echo "Architecture: $ARCH"

# ── Python venv ──
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi
source venv/bin/activate

# ── PyTorch CUDA ──
echo "Installing PyTorch CUDA..."
pip install --upgrade pip

# ARM64 (DGX): use NVIDIA's PyTorch index
if [ "$ARCH" = "aarch64" ]; then
    echo "→ ARM64 detected — installing PyTorch from NVIDIA index"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "→ x86_64 detected — installing PyTorch from default index"
    pip install torch torchvision torchaudio
fi

# ── VoxCPM2 ──
echo "Installing voxcpm + dependencies..."
pip install voxcpm fastapi uvicorn soundfile numpy pydantic

# ── Verify ──
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test VoxCPM2:"

if [ "$ARCH" = "aarch64" ]; then
    echo "  TORCHDYNAMO_DISABLE=1 python server.py"
    echo ""
    echo "  ARM64 needs TORCHDYNAMO_DISABLE=1"
    echo "  (Triton JIT not yet supported on aarch64)"
else
    echo "  python server.py"
fi

echo ""
echo "Health check: curl http://localhost:8808/health"
