"""Build the codec evaluation matrix: re-encode dev/eval audio through real codecs.

For every condition in src/codec.CODEC_MATRIX (except clean) this:
  1. re-encodes each flac with ffmpeg (true codec round-trip)
  2. writes a float16 waveform cache  data/processed/cm_{cond}_{split}_waveforms.npy
  3. writes mel shards                data/processed/cm_{cond}_{split}_shards/
  4. deletes the re-encoded audio + waveform cache unless --keep_audio

Resumable: --conditions, --start/--end, and shard-exists checks make this safe
to run across multiple Kaggle sessions (the full matrix is ~780k ffmpeg calls,
roughly 5-6 h on a 4-core session for eval).
"""

import os
import sys
import shutil
import argparse
import tempfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.codec import CODEC_MATRIX, get_ffmpeg
from src.protocol import CM_PROTOCOLS, CM_AUDIO_DIRS, load_cm_protocol
from src.dataset import build_waveforms, build_clean_mel_shards, PROCESSED


def _reencode_one(job):
    src, dst, ffmpeg, codec, bitrate = job
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        return  # resumable
    enc_ext = {"libopus": ".ogg", "libmp3lame": ".mp3", "aac": ".m4a", "libopencore_amrnb": ".amr"}[codec]
    import subprocess

    def run(cmd):
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {src}: {proc.stderr.decode()[-300:]}")

    with tempfile.TemporaryDirectory() as tmp:
        enc = os.path.join(tmp, "enc" + enc_ext)
        wav = os.path.join(tmp, "dec.wav")
        if codec == "libopencore_amrnb":
            run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", "8000", "-ac", "1",
                 "-c:a", codec, "-b:a", bitrate, enc])
        else:
            run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1",
                 "-c:a", codec, "-b:a", bitrate, enc])
        run([ffmpeg, "-y", "-loglevel", "error", "-i", enc, "-ar", "16000", "-ac", "1", wav])
        # copy with correct length guard
        import librosa
        audio, _ = librosa.load(wav, sr=16000, mono=True)
        target_len = sf.info(src).frames
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        sf.write(dst, audio.astype(np.float32), 16000, subtype="PCM_16", format="WAV")


def build_condition(cond, split, workers, start, end, keep_audio, out_root=None):
    codec_spec = CODEC_MATRIX[cond]
    if codec_spec is None:
        print(f"{cond}: clean — nothing to build")
        return
    codec, bitrate = codec_spec
    ffmpeg = get_ffmpeg()
    ids, _, _ = load_cm_protocol(CM_PROTOCOLS[split])
    src_dir = CM_AUDIO_DIRS[split]
    out_root = out_root or os.path.join(PROCESSED, f"codec_matrix", cond, split)
    audio_dir = os.path.join(out_root, "flac")
    os.makedirs(audio_dir, exist_ok=True)

    jobs = []
    for i in range(start, (end or len(ids))):
        fid = ids[i]
        dst = os.path.join(audio_dir, f"{fid}.wav")
        src = os.path.join(src_dir, f"{fid}.flac")
        jobs.append((src, dst, ffmpeg, codec, bitrate))

    with Pool(workers) as p:
        for n, _ in enumerate(p.imap_unordered(_reencode_one, jobs, chunksize=16)):
            if (n + 1) % 2000 == 0:
                print(f"  [{cond}/{split}] reencoded {n + 1}/{len(jobs)}", flush=True)

    wf_path = os.path.join(PROCESSED, f"cm_{cond}_{split}_waveforms.npy")
    shard_out = os.path.join(PROCESSED, f"cm_{cond}_{split}_shards")
    build_waveforms(split, workers=workers, audio_dir=audio_dir, out_path=wf_path)
    build_clean_mel_shards(split, workers=workers, cache_path=wf_path, out_dir=shard_out)

    if not keep_audio:
        shutil.rmtree(out_root, ignore_errors=True)
        if os.path.exists(wf_path):
            os.remove(wf_path)
    print(f"[{cond}/{split}] shards -> {shard_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="subset of: " + ", ".join(k for k, v in CODEC_MATRIX.items() if v))
    parser.add_argument("--split", default="eval", choices=["dev", "eval"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--keep_audio", action="store_true")
    args = parser.parse_args()

    conds = args.conditions or [k for k, v in CODEC_MATRIX.items() if v is not None]
    for cond in conds:
        print(f"=== {cond} ({args.split}) ===")
        build_condition(cond, args.split, args.workers, args.start, args.end, args.keep_audio)
