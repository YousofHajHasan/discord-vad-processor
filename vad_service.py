"""
VAD Processor Service  (incremental / always-running)
──────────────────────────────────────────────────────
Watches Recordings/ for new or grown daily MP3 files, then:

  1. Runs Silero VAD only on the NEW audio tail since last run
     - Processes in WINDOW_MINUTES windows to keep RAM flat
     - State stores, per user+date: "processed_samples" (int)
     - On next scan: seek into file, run VAD on the rest,
       append chunk_NNN.wav files numbered after existing ones
  2. Loops forever with POLL_INTERVAL sleep

Chunk filenames reflect the order they were created, so they are
always sorted chronologically when listed alphabetically:
    chunk_001.wav  -> first speech segment ever found
    chunk_002.wav  -> second, etc.
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
from mutagen.mp3 import MP3

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
POLL_INTERVAL    = int(os.environ.get("POLL_INTERVAL",     "30"))
STABILITY_WINDOW = int(os.environ.get("STABILITY_WINDOW",  "30"))

SAMPLE_RATE      = 16000
VAD_THRESHOLD    = float(os.environ.get("VAD_THRESHOLD",   "0.5"))
SILENCE_SEC      = float(os.environ.get("SILENCE_SEC",     "1.0"))
MIN_SPEECH_MS    = int(os.environ.get("MIN_SPEECH_MS",     "250"))
SPEECH_PAD_MS    = int(os.environ.get("SPEECH_PAD_MS",     "30"))

WINDOW_MINUTES   = int(os.environ.get("WINDOW_MINUTES",    "10"))
WINDOW_SECONDS   = WINDOW_MINUTES * 60


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


# -- Audio helpers ------------------------------------------------------------
def load_audio_ffmpeg(path: Path, start_sec: float = 0.0,
                      duration_sec: float = None) -> np.ndarray:
    """
    Decode audio to float32 mono 16kHz.
    start_sec    — seek to this position before decoding
    duration_sec — decode at most this many seconds (None = to end of file)
    """
    cmd = ["ffmpeg", "-y"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(path)]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-loglevel", "quiet", "pipe:1",
    ]
    raw = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def get_file_duration_sec(path: Path) -> float:
    """
    Read MP3 duration from header metadata only — no seeking, no decoding.
    Replaces the old ffprobe-based implementation that caused ~1TB/day of
    unnecessary disk reads by seeking through the entire file on every call.
    """
    try:
        return MP3(path).info.length
    except Exception:
        return 0.0


def is_stable(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) >= STABILITY_WINDOW
    except FileNotFoundError:
        return False


# -- State tracking -----------------------------------------------------------
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


def state_key(user_dir: Path, date_str: str) -> str:
    return f"{user_dir.name}:{date_str}"


# -- Incremental VAD processing (windowed) ------------------------------------
def process_incremental(daily_file: Path, user_dir: Path, date_str: str,
                        vad: SileroVAD, state: dict) -> bool:
    """
    Processes only the new audio tail since the last run, in WINDOW_SECONDS
    windows to keep peak RAM usage flat regardless of file size.

    For each window:
      1. Decode WINDOW_SECONDS of audio starting at current offset
      2. Run VAD, write any speech chunks to disk
      3. Advance state by the window size
      4. Save state immediately — crash-safe per window

    Returns True if any state was modified.
    """
    key   = state_key(user_dir, date_str)
    entry = state.get(key, {"processed_samples": 0, "chunk_count": 0})

    prev_samples = int(entry.get("processed_samples", 0))
    prev_chunks  = int(entry.get("chunk_count", 0))
    prev_sec     = prev_samples / SAMPLE_RATE

    # Read duration from MP3 headers only (~10KB), not by seeking the file
    total_sec     = get_file_duration_sec(daily_file)
    total_samples = int(total_sec * SAMPLE_RATE)

    if total_samples <= prev_samples:
        log.debug(
            f"  [{user_dir.name}/{date_str}] "
            f"No new audio (total={total_sec:.1f}s, done={prev_sec:.1f}s)"
        )
        return False

    new_seconds = total_sec - prev_sec
    log.info(
        f"  [{user_dir.name}/{date_str}] "
        f"+{new_seconds:.1f}s new audio to process "
        f"(total {total_sec:.1f}s, already done {prev_sec:.1f}s) "
        f"in {int(np.ceil(new_seconds / WINDOW_SECONDS))} window(s) "
        f"of {WINDOW_MINUTES}min each"
    )

    chunks_dir  = user_dir / "chunks" / date_str
    current_sec = prev_sec
    chunk_count = prev_chunks
    changed     = False

    while current_sec < total_sec:
        window_dur = min(WINDOW_SECONDS, total_sec - current_sec)

        try:
            window_audio = load_audio_ffmpeg(
                daily_file,
                start_sec=current_sec,
                duration_sec=window_dur,
            )
        except Exception as e:
            log.error(f"  Failed to decode window at {current_sec:.1f}s: {e}")
            break

        if len(window_audio) == 0:
            break

        timestamps = get_speech_timestamps(window_audio, vad)

        if timestamps:
            chunks_dir.mkdir(parents=True, exist_ok=True)
            for i, ts in enumerate(timestamps):
                chunk    = window_audio[ts["start"] : ts["end"]]
                idx      = chunk_count + i + 1
                out_path = chunks_dir / f"chunk_{idx:03d}.wav"
                sf.write(str(out_path), chunk, SAMPLE_RATE)

            new_count = chunk_count + len(timestamps)
            log.info(
                f"  [{user_dir.name}/{date_str}] "
                f"window {current_sec:.0f}s-{current_sec+window_dur:.0f}s: "
                f"{len(timestamps)} segment(s) -> "
                f"chunk_{chunk_count+1:03d}.wav ... chunk_{new_count:03d}.wav"
            )
            chunk_count = new_count
        else:
            log.debug(
                f"  [{user_dir.name}/{date_str}] "
                f"window {current_sec:.0f}s-{current_sec+window_dur:.0f}s: "
                f"no speech"
            )

        current_sec += len(window_audio) / SAMPLE_RATE

        state[key] = {
            "processed_samples": int(current_sec * SAMPLE_RATE),
            "chunk_count":       chunk_count,
            "last_run":          datetime.utcnow().isoformat(),
        }
        save_state(state)
        changed = True

        del window_audio

    return changed


# -- Main watchdog loop -------------------------------------------------------
def scan_once(vad: SileroVAD, state: dict) -> bool:
    changed = False

    if not RECORDINGS_DIR.exists():
        log.warning(f"Recordings dir not found: {RECORDINGS_DIR}")
        return False

    for user_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        if not user_dir.name.isdigit():
            continue

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
    log.info(f"  Decode window   : {WINDOW_MINUTES}min ({WINDOW_SECONDS}s)")

    if not ONNX_MODEL.exists():
        log.error(f"ONNX model not found at {ONNX_MODEL}. Exiting.")
        return

    vad   = SileroVAD(ONNX_MODEL)
    state = load_state()
    log.info(f"Loaded state: {len(state)} entry/entries.")

    while True:
        try:
            scan_once(vad, state)
        except Exception as e:
            log.error(f"Scan error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()