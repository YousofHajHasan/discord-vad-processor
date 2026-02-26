"""
VAD Processor Service  (incremental / always-running)
──────────────────────────────────────────────────────
Watches Recordings/ for new or grown daily MP3 files, then:

  1. Converts .pcm -> .mp3
  2. Merges *_part*.mp3 into YYYY-MM-DD.mp3
  3. Runs Silero VAD only on the NEW audio tail since last run
     - State stores, per user+date: "processed_samples" (int)
     - On next scan: decode full file, skip the first N samples,
       run VAD on the rest, append chunk_NNN.wav files numbered
       after the existing ones
  4. Loops forever with POLL_INTERVAL sleep

Chunk filenames reflect the order they were created, so they are
always sorted chronologically when listed alphabetically:
    chunk_001.wav  -> first speech segment ever found
    chunk_002.wav  -> second, etc.
    (new ones appended: chunk_006.wav, chunk_007.wav ...)
"""

import os
import json
import time
import subprocess
import logging
import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
from datetime import datetime

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VAD] %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# -- Configuration ------------------------------------------------------------
RECORDINGS_DIR   = Path(os.environ.get("RECORDINGS_PATH",  "/app/recordings"))
ONNX_MODEL       = Path(os.environ.get("ONNX_MODEL_PATH",  "/app/silero_vad.onnx"))
STATE_FILE       = Path(os.environ.get("STATE_FILE",       "/app/processed.json"))
POLL_INTERVAL    = int(os.environ.get("POLL_INTERVAL",    "30"))
STABILITY_WINDOW = int(os.environ.get("STABILITY_WINDOW", "30"))

SAMPLE_RATE      = 16000
VAD_THRESHOLD    = float(os.environ.get("VAD_THRESHOLD",  "0.5"))
SILENCE_SEC      = float(os.environ.get("SILENCE_SEC",    "1.0"))
MIN_SPEECH_MS    = int(os.environ.get("MIN_SPEECH_MS",    "250"))
SPEECH_PAD_MS    = int(os.environ.get("SPEECH_PAD_MS",    "30"))


# -- Silero VAD ONNX wrapper --------------------------------------------------
class SileroVAD:
    WINDOW  = 512
    CONTEXT = 64

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset_states()

    def reset_states(self):
        self._state   = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        x  = chunk.reshape(1, -1).astype(np.float32)
        x  = np.concatenate([self._context, x], axis=1)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        out, self._state = self.session.run(
            None, {"input": x, "sr": sr, "state": self._state},
        )
        self._context = x[:, -self.CONTEXT:]
        return float(out[0][0])


def get_speech_timestamps(audio: np.ndarray, vad: SileroVAD) -> list:
    """Returns list of {"start", "end"} sample indices relative to the given audio slice."""
    WINDOW              = SileroVAD.WINDOW
    min_speech_samples  = int(SAMPLE_RATE * MIN_SPEECH_MS / 1000)
    min_silence_samples = int(SAMPLE_RATE * SILENCE_SEC)
    speech_pad_samples  = int(SAMPLE_RATE * SPEECH_PAD_MS / 1000)
    neg_threshold       = max(VAD_THRESHOLD - 0.15, 0.01)
    audio_len           = len(audio)

    vad.reset_states()
    triggered = False
    speeches  = []
    current   = {}
    temp_end  = 0

    for i in range(0, audio_len, WINDOW):
        chunk = audio[i : i + WINDOW]
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
        prob = vad(chunk)

        if prob >= VAD_THRESHOLD:
            if temp_end:
                temp_end = 0
            if not triggered:
                triggered = True
                current["start"] = i
        elif prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = i
            if (i - temp_end) >= min_silence_samples:
                current["end"] = temp_end
                if (current["end"] - current["start"]) >= min_speech_samples:
                    speeches.append(current)
                current   = {}
                triggered = False
                temp_end  = 0

    if triggered and current:
        current["end"] = audio_len
        if (current["end"] - current["start"]) >= min_speech_samples:
            speeches.append(current)

    for s in speeches:
        s["start"] = max(0, s["start"] - speech_pad_samples)
        s["end"]   = min(audio_len, s["end"] + speech_pad_samples)

    return speeches


# -- Audio helpers -------------------------------------------------------------
def load_audio_ffmpeg(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def pcm_to_mp3(pcm_path: Path) -> Path:
    mp3_path = pcm_path.with_suffix(".mp3")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", "48000", "-ac", "2",
        "-i", str(pcm_path),
        "-loglevel", "error",
        str(mp3_path),
    ])
    if result.returncode == 0 and mp3_path.exists():
        pcm_path.unlink()
        log.info(f"  Converted PCM -> {mp3_path.name}")
        return mp3_path
    log.warning(f"  Failed to convert {pcm_path.name}")
    return None


def merge_parts(folder: Path, date_str: str) -> Path:
    """Merge stable *_part*.mp3 (+ any existing daily file) into YYYY-MM-DD.mp3."""
    parts = sorted(folder.glob("*_part*.mp3"))
    if not parts:
        return None

    daily_file      = folder / f"{date_str}.mp3"
    list_file       = folder / "_merge_list.txt"
    files_to_concat = []

    if daily_file.exists():
        temp = folder / f"_temp_{date_str}.mp3"
        daily_file.rename(temp)
        files_to_concat.append(temp)
    else:
        temp = None

    files_to_concat.extend(parts)

    with open(list_file, "w") as f:
        for p in files_to_concat:
            f.write(f"file '{p.resolve()}'\n")

    log.info(f"  Merging {len(parts)} part(s) -> {daily_file.name}")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(daily_file),
        "-loglevel", "error",
    ], check=False)

    list_file.unlink(missing_ok=True)
    if temp and temp.exists():
        temp.unlink()
    for p in parts:
        p.unlink(missing_ok=True)

    return daily_file if daily_file.exists() else None


# -- State tracking ------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_stable(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) >= STABILITY_WINDOW
    except FileNotFoundError:
        return False


def state_key(user_dir: Path, date_str: str) -> str:
    return f"{user_dir.name}:{date_str}"


# -- Incremental VAD processing ------------------------------------------------
def process_incremental(daily_file: Path, user_dir: Path, date_str: str,
                         vad: SileroVAD, state: dict) -> bool:
    """
    Decode full daily file, skip already-processed samples, run VAD on the
    new tail only, append chunk_NNN.wav files numbered after existing ones.
    Returns True if state was modified.
    """
    key   = state_key(user_dir, date_str)
    entry = state.get(key, {"processed_samples": 0, "chunk_count": 0})

    prev_samples = int(entry.get("processed_samples", 0))
    prev_chunks  = int(entry.get("chunk_count", 0))

    try:
        audio = load_audio_ffmpeg(daily_file)
    except Exception as e:
        log.error(f"  Failed to decode {daily_file}: {e}")
        return False

    total_samples = len(audio)

    if total_samples <= prev_samples:
        log.debug(
            f"  [{user_dir.name}/{date_str}] "
            f"No new audio (total={total_samples}, done={prev_samples})"
        )
        return False

    new_seconds = (total_samples - prev_samples) / SAMPLE_RATE
    log.info(
        f"  [{user_dir.name}/{date_str}] "
        f"+{new_seconds:.1f}s new audio  "
        f"(total {total_samples/SAMPLE_RATE:.1f}s, "
        f"already processed {prev_samples/SAMPLE_RATE:.1f}s)"
    )

    new_audio  = audio[prev_samples:]
    timestamps = get_speech_timestamps(new_audio, vad)
    log.info(f"  Found {len(timestamps)} new speech segment(s).")

    # Always advance the offset, even if no speech was found
    new_entry = {
        "processed_samples": total_samples,
        "chunk_count": prev_chunks,
        "last_run": datetime.utcnow().isoformat(),
    }

    if timestamps:
        chunks_dir = user_dir / "chunks" / date_str
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, ts in enumerate(timestamps):
            chunk    = new_audio[ts["start"] : ts["end"]]
            idx      = prev_chunks + i + 1
            out_path = chunks_dir / f"chunk_{idx:03d}.wav"
            sf.write(str(out_path), chunk, SAMPLE_RATE)

        new_chunk_count = prev_chunks + len(timestamps)
        log.info(
            f"  Saved chunk_{prev_chunks+1:03d}.wav ... chunk_{new_chunk_count:03d}.wav"
            f"  -> {chunks_dir}"
        )
        new_entry["chunk_count"] = new_chunk_count

    state[key] = new_entry
    return True


# -- Main watchdog loop --------------------------------------------------------
def scan_once(vad: SileroVAD, state: dict) -> bool:
    changed = False

    if not RECORDINGS_DIR.exists():
        log.warning(f"Recordings dir not found: {RECORDINGS_DIR}")
        return False

    for user_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not user_dir.is_dir():
            continue

        # 1. Convert leftover .pcm files
        for pcm in user_dir.glob("*.pcm"):
            if is_stable(pcm):
                pcm_to_mp3(pcm)

        # 2. Merge stable _part*.mp3 -> today's daily file
        stable_parts = [p for p in sorted(user_dir.glob("*_part*.mp3")) if is_stable(p)]
        if stable_parts:
            date_str = datetime.now().strftime("%Y-%m-%d")
            merged   = merge_parts(user_dir, date_str)
            if merged:
                log.info(f"[{user_dir.name}] Merged parts -> {merged.name}")
                changed = True

        # 3. Incremental VAD on every daily MP3
        for daily_file in sorted(user_dir.glob("????-??-??.mp3")):
            if not is_stable(daily_file):
                log.debug(f"  Skipping {daily_file.name} (still being written)")
                continue
            date_str = daily_file.stem
            try:
                if process_incremental(daily_file, user_dir, date_str, vad, state):
                    changed = True
            except Exception as e:
                log.error(f"[{user_dir.name}] Error on {daily_file.name}: {e}")

    return changed


def main():
    log.info("VAD Processor starting up (incremental mode)...")
    log.info(f"  Recordings dir  : {RECORDINGS_DIR}")
    log.info(f"  ONNX model      : {ONNX_MODEL}")
    log.info(f"  Poll interval   : {POLL_INTERVAL}s")
    log.info(f"  Stability window: {STABILITY_WINDOW}s")
    log.info(f"  VAD threshold   : {VAD_THRESHOLD}")
    log.info(f"  Silence gap     : {SILENCE_SEC}s")

    if not ONNX_MODEL.exists():
        log.error(f"ONNX model not found at {ONNX_MODEL}. Exiting.")
        return

    vad   = SileroVAD(ONNX_MODEL)
    state = load_state()
    log.info(f"Loaded state: {len(state)} entry/entries.")

    while True:
        try:
            changed = scan_once(vad, state)
            if changed:
                save_state(state)
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
