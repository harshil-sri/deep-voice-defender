"""True codec simulation: round-trip audio through real speech codecs via ffmpeg.

This replaces the v1 "fake Opus" (16k->8k->16k resampling). Real encoding
introduces the actual codec's quantization, bandwidth extension and packetization
artifacts, which is the phenomenon the paper claims to study.

Binary discovery order: $FFMPEG_BIN -> `ffmpeg` on PATH -> imageio_ffmpeg static
binary (pip install imageio-ffmpeg). Kaggle images ship ffmpeg on PATH.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

OPUS_BITRATES = ["6000", "12000", "24000", "48000"]  # bits/s, supported by libopus
MP3_BITRATES = ["16k", "32k", "64k"]
AAC_BITRATES = ["16k", "32k", "64k"]
AMRNB_BITRATES = ["4.75k", "7.4k", "12.2k"]  # AMR-NB modes (8 kHz narrowband)

# Named conditions for the evaluation matrix (see scripts/build_codec_matrix.py)
CODEC_MATRIX = {
    "clean": None,
    "opus_6k": ("libopus", "6000"),
    "opus_12k": ("libopus", "12000"),
    "opus_24k": ("libopus", "24000"),
    "opus_48k": ("libopus", "48000"),
    "mp3_16k": ("libmp3lame", "16k"),
    "mp3_32k": ("libmp3lame", "32k"),
    "mp3_64k": ("libmp3lame", "64k"),
    "aac_16k": ("aac", "16k"),
    "aac_32k": ("aac", "32k"),
    "amrnb_4750": ("libopencore_amrnb", "4.75k"),
    "amrnb_12200": ("libopencore_amrnb", "12.2k"),
}

_ffmpeg_cache = None


def get_ffmpeg():
    global _ffmpeg_cache
    if _ffmpeg_cache is None:
        _ffmpeg_cache = (
            os.environ.get("FFMPEG_BIN")
            or shutil.which("ffmpeg")
            or _imageio_ffmpeg()
        )
    if _ffmpeg_cache is None:
        raise RuntimeError(
            "ffmpeg not found. Set FFMPEG_BIN or `pip install imageio-ffmpeg`."
        )
    return _ffmpeg_cache


def _imageio_ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr.decode()[-800:]}")


# ffmpeg infers the muxer from the file extension, so each codec gets its own container
CODEC_CONTAINER = {
    "libopus": ".ogg",
    "libmp3lame": ".mp3",
    "aac": ".m4a",
    "libopencore_amrnb": ".amr",
}


def codec_roundtrip(waveform, codec, bitrate, sr=16000):
    """Encode `waveform` with the given codec and decode it back to float32 @ sr.

    codec: ffmpeg encoder name (libopus, libmp3lame, aac, libopencore_amrnb).
    Length of the output is forced back to the input length (codecs/padding can
    shift it by a few samples).
    """
    from src.preprocessing import pad_or_trim  # local import avoids cycles

    waveform = np.asarray(waveform, dtype=np.float32)
    n_in = len(waveform)
    ffmpeg = get_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        enc = os.path.join(tmp, "enc" + CODEC_CONTAINER[codec])
        out = os.path.join(tmp, "out.wav")
        sf.write(src, waveform, sr, subtype="PCM_16")

        if codec == "libopencore_amrnb":
            # AMR-NB is narrowband: resample to 8 kHz before encoding, back after.
            _run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", "8000", "-ac", "1",
                  "-c:a", codec, "-b:a", bitrate, enc])
        else:
            _run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", str(sr), "-ac", "1",
                  "-c:a", codec, "-b:a", bitrate, enc])
            if os.path.getsize(enc) == 0:
                raise RuntimeError(f"{codec} produced an empty file")
        _run([ffmpeg, "-y", "-loglevel", "error", "-i", enc, "-ar", str(sr), "-ac", "1", out])

        decoded, _ = sf.read(out, dtype="float32")
    decoded = pad_or_trim(decoded, n_in)
    return decoded


def random_codec_degrade(waveform, rng, sr=16000):
    """Training-time codec augmentation: one random codec at one random bitrate."""
    codec = rng.choice(["libopus", "libmp3lame", "aac"])
    if codec == "libopus":
        bitrate = rng.choice(OPUS_BITRATES)
    elif codec == "libmp3lame":
        bitrate = rng.choice(MP3_BITRATES)
    else:
        bitrate = rng.choice(AAC_BITRATES)
    return codec_roundtrip(waveform, codec, str(bitrate), sr=sr)
