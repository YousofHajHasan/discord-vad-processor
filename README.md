# discord-vad-processor

Standalone Docker service that watches the shared `recordings/` volume and:

1. Converts any leftover `.pcm` files → `.mp3`
2. Merges `_part*.mp3` chunks → `YYYY-MM-DD.mp3` (daily file per user)
3. Runs **Silero VAD** (ONNX, no PyTorch) to split each daily file into speech chunks saved at:
   ```
   <user_folder>/chunks/<YYYY-MM-DD>/chunk_001.wav
   <user_folder>/chunks/<YYYY-MM-DD>/chunk_002.wav
   ...
   ```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RECORDINGS_PATH` | `/app/recordings` | Shared volume path |
| `ONNX_MODEL_PATH` | `/app/silero_vad.onnx` | Path to Silero VAD ONNX model |
| `STATE_FILE` | `/app/processed.json` | Tracks already-processed files |
| `POLL_INTERVAL` | `30` | Seconds between folder scans |
| `STABILITY_WINDOW` | `30` | Seconds a file must be unmodified before processing |
| `VAD_THRESHOLD` | `0.5` | Speech confidence threshold (0–1) |
| `SILENCE_SEC` | `1.0` | Minimum silence gap to split on (seconds) |
| `MIN_SPEECH_MS` | `250` | Minimum speech duration to keep (ms) |
| `SPEECH_PAD_MS` | `30` | Padding added to each chunk on both sides (ms) |

## Building

```bash
docker build -t discord-vad-processor .
```

> **Note:** `silero_vad.onnx` must be present in the build context.  
> Download from: https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
