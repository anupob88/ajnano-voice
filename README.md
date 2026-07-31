# ajnano-voice

VoxCPM2 Thai voice synthesis for ajnano framework — real-time TTS with zero-shot voice cloning.

## Overview

Voice bridge server connecting Oracle agents (Nova, Tonkan, Tonhop) to real-time speech synthesis. Migrating from F5-TTS to VoxCPM2 for better Thai support (85.4% voice similarity) and production-grade serving via Nano-vLLM.

## Quick Start

```bash
# Install VoxCPM2
pip install voxcpm

# Generate speech (Thai)
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
wav = model.generate(
    text="สวัสดีครับ ผมต้นกัญ ยินดีที่ได้รู้จัก",
    prompt_wav_path="ref_tonkan.wav",  # optional: voice cloning
)
```

See [ROADMAP.md](ROADMAP.md) for full migration plan.

## Architecture

- **TTS Engine**: VoxCPM2 (2B params, Apache 2.0, 30 languages incl. Thai)
- **Serving**: Nano-vLLM (RTF ~0.13 on RTX 4090) or FastAPI wrapper
- **Deploy**: 192.168.1.121 (RTX 3080 Ti 12GB)
- **Voice Server**: `ajnano-voice-server.py` (WebSocket + HTTP, Python/aiohttp)

## License

Apache 2.0 — matching VoxCPM2 upstream license.
