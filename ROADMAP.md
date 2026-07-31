# VoxCPM2 Migration Roadmap

Migrate ajnano voice bridge from F5-TTS → VoxCPM2.

## Phase 1: Setup & Validation (Nova)

- [x] Create GitHub repo `ajnano-voice`
- [ ] Install VoxCPM2 on .121 (RTX 3080 Ti 12GB)
  ```bash
  pip install voxcpm
  ```
- [ ] Download VoxCPM2 weights (~2B params, ~4GB)
- [ ] Test Thai generation — assess quality, WER/CER
- [ ] Test zero-shot voice cloning with Thai reference audio
- [ ] Benchmark RTF on 3080 Ti

## Phase 2: API Server (Nova)

- [ ] Deploy Nano-vLLM or FastAPI wrapper on .121
- [ ] OpenAI-compatible `/v1/audio/speech` endpoint
- [ ] Streaming support (chunk-by-chunk)
- [ ] Multi-voice support (cloned voices for each Oracle agent)

## Phase 3: Integration (Tonkan)

- [ ] Update `ajnano-voice-server.py` — replace F5-TTS with VoxCPM2 API calls
- [ ] Create voice profiles for Oracle agents:
  - Nova voice clone
  - Tonkan voice clone
  - Tonhop voice clone
- [ ] WebSocket streaming support
- [ ] Fallback to F5-TTS if VoxCPM2 unavailable

## Phase 4: Production

- [ ] Voice Bridge → Discord voice channel integration
- [ ] FarmBot voice integration
- [ ] MooYor Telegram voice
- [ ] Performance monitoring & auto-scaling

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Discord/Web │────▶│ ajnano-voice     │────▶│ VoxCPM2 (.121)  │
│  Clients     │◀────│ server (.43)     │◀────│ :8082 (TTS API) │
└──────────────┘     └──────────────────┘     └─────────────────┘
       │                      │                        │
       ▼                      ▼                        ▼
   Audio input           STT (port 5004)          Nano-vLLM
   (mic/upload)          CCB brain                RTX 3080 Ti
```

## Reference

- VoxCPM2: https://github.com/OpenBMB/VoxCPM
- Current F5-TTS server: `ajnano-voice-server.py`
- Deploy machine: 192.168.1.121 (admin), E:\Projects\ajnano-voice\
