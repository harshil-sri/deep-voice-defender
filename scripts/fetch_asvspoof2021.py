#!/usr/bin/env python3
"""Fetch the ASVspoof 2021 LA evaluation data.

What lives where (verified 2026-10-23):

* LA eval AUDIO (181,566 flac trials)  -> Zenodo record 4837263
  (DOI 10.5281/zenodo.4837263, "ASVspoof 2021 Challenge - Logical Access
  Database" v1.1), single file ASVspoof2021_LA_eval.tar.gz (~7.8 GB, md5
  2abee34d8b0b91159555fc4f016e4562). The tarball embeds a BARE trial list
  (ASVspoof2021.LA.cm.eval.trl.txt, one utterance id per line, md5
  2dfda95bec4facf6e4bde655dcffd8a7) and the README/LICENSE - but NO labels.

* Eval KEY-LABELS + meta-labels (codec / transducer / attack per trial) ->
  NOT on Zenodo. Published post-challenge by the organizers at
  https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
  (md5 f052cc2ed276745afa3b5198665d3b26). Untars to keys/LA/CM/trial_metadata.txt
  (8-column CM protocol WITH labels), keys/LA/ASV/trial_metadata.txt and the
  organizers' official ASV scores keys/LA/ASV/ASVTorch_Kaldi/score.txt
  (needed for official min t-DCF) plus pre-computed t-DCF C012 coefficients.
  NOTE: asvspoof.org intermittently (at times persistently) answers 503; the
  official download.sh loops until it succeeds. This script retries too.

* Kaggle mirror (mohammedabdeldayem/avsspoof-2021) contains the full LA eval
  partition (flac + the same bare trl list + README/LICENSE) plus at least
  DF eval part00; it does NOT contain the keys/labels/ASV scores. Supported
  via `--source kaggle` (requires the `kaggle` CLI + API token) as a fallback
  for the audio.

* Ground-truth labels only (no audio): HuggingFace mirror
  SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA ships data/labels.parquet
  (utterance_id + label, 181,566 rows) built from the official keys; fetched
  with `--labels-hf` when asvspoof.org is unreachable.

Examples
--------
    # audio + protocol list + best-effort keys (default destination)
    python scripts/fetch_asvspoof2021.py

    # protocols / keys / ASV scores only (no 7.8 GB audio)
    python scripts/fetch_asvspoof2021.py --only-protocols

    # also grab the label parquet from the HF mirror as a backup
    python scripts/fetch_asvspoof2021.py --only-protocols --labels-hf

    # audio from the Kaggle mirror instead of Zenodo
    python scripts/fetch_asvspoof2021.py --source kaggle

    # sanity-check what is already on disk
    python scripts/fetch_asvspoof2021.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Verified constants
# --------------------------------------------------------------------------
DEFAULT_RECORD_ID = "4837263"  # Zenodo: ASVspoof 2021 Logical Access (LA) eval
ZENODO_API = "https://zenodo.org/api/records/{record_id}"

AUDIO_ARCHIVE_MD5 = "2abee34d8b0b91159555fc4f016e4562"  # ASVspoof2021_LA_eval.tar.gz
TRL_LIST_MD5 = "2dfda95bec4facf6e4bde655dcffd8a7"  # ASVspoof2021.LA.cm.eval.trl.txt
N_EVAL_TRIALS = 181566
N_BONAFIDE = 18452
N_SPOOF = 163114

KEYS_URL = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
KEYS_MD5 = "f052cc2ed276745afa3b5198665d3b26"

# Verified mirror of the bare (label-free) challenge-time eval trial list.
TRL_FALLBACK_URL = (
    "https://raw.githubusercontent.com/TakHemlata/SSL_Anti-spoofing/main/"
    "database/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt"
)
# Label-only parquet (utterance_id, label) built from the official key file.
HF_LABELS_URL = (
    "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/"
    "ASVspoof2021_LA/resolve/main/data/labels.parquet"
)
KAGGLE_DATASET = "mohammedabdeldayem/avsspoof-2021"

CHUNK = 1024 * 1024


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, dest: str, retries: int = 8, retry_wait: float = 10.0) -> str:
    """Download `url` to `dest`, resumable via Range + a `.part` temp file.

    The official ASVspoof server answers 503 "busy" a lot (the organizers'
    own download.sh loops on it), so failures are retried with backoff.
    Returns `dest`. If the server ignores Range and restarts, we restart too.
    """
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        existing = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "fetch_asvspoof2021/1.0"})
        if existing:
            req.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.getcode()
                if status == 206:
                    total = existing + int(resp.headers["Content-Length"])
                    mode, start = "ab", existing
                elif status == 200:
                    total = int(resp.headers["Content-Length"])
                    mode, start = "wb", 0
                else:
                    raise urllib.error.HTTPError(url, status, "unexpected status", resp.headers, None)
                if existing and mode == "wb":
                    print(f"  server ignored Range; restarting from 0")
                done = start
                if total:
                    print(f"  resuming/starting at {start}/{total} bytes ({100*start/max(total,1):.1f}%)")
                with open(part, mode) as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total and done % (100 * CHUNK) < CHUNK:
                            print(f"  {done}/{total} ({100*done/total:.1f}%)", flush=True)
            if total and done != total:
                raise IOError(f"truncated download: {done} != {total}")
            os.replace(part, dest)
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, IOError) as exc:
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(retry_wait * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def http_download_small(url: str, dest: str, retries: int = 3) -> str:
    """Plain (non-resumable) download for small files."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fetch_asvspoof2021/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


# --------------------------------------------------------------------------
# Zenodo
# --------------------------------------------------------------------------
def zenodo_files(record_id: str) -> list[dict]:
    url = ZENODO_API.format(record_id=record_id)
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_asvspoof2021/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        import json

        meta = json.loads(resp.read().decode("utf-8"))
    return meta.get("files", [])


def download_audio_from_zenodo(dest: str, record_id: str, extract: bool) -> str:
    files = zenodo_files(record_id)
    if not files:
        raise RuntimeError(f"record {record_id} has no files")
    os.makedirs(dest, exist_ok=True)
    for finfo in files:
        key, link = finfo["key"], finfo["links"][0]
        size, checksum = finfo.get("size"), (finfo.get("checksum") or "").removeprefix("md5:")
        out = os.path.join(dest, key)
        if os.path.exists(out):
            print(f"[zenodo] {key} already present ({size} bytes expected)")
        else:
            print(f"[zenodo] downloading {key} ({size} bytes)")
            http_get(link, out)
        if checksum:
            got = md5_of_file(out)
            if got != checksum:
                raise RuntimeError(f"md5 mismatch for {key}: got {got}, expected {checksum}")
            print(f"[zenodo] {key}: md5 OK")
        if extract and tarfile.is_tarfile(out):
            print(f"[zenodo] extracting {key}")
            with tarfile.open(out) as tar:
                tar.extractall(dest)
    return dest


# --------------------------------------------------------------------------
# asvspoof.org keys (labels + ASV scores)
# --------------------------------------------------------------------------
def download_keys(dest: str, retries: int = 20) -> str | None:
    """Download LA-keys-full.tar.gz (trial_metadata keys, meta-labels, official
    ASV scores, C012 coeffs). Returns the archive path, or None if the server
    stays unreachable (503)."""
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, os.path.basename(KEYS_URL))
    if os.path.exists(out):
        print(f"[keys] {out} already present")
    else:
        print(f"[keys] trying {KEYS_URL} (server is often busy; retrying)")
        try:
            http_get(KEYS_URL, out, retries=retries, retry_wait=20.0)
        except RuntimeError as exc:
            print(f"[keys] FAILED: {exc}")
            print("[keys] asvspoof.org is currently unreachable (503).")
            print("[keys] Eval labels will be MISSING. Alternatives:")
            print(f"[keys]   1. re-run later / with more retries (official source)")
            print(f"[keys]   2. --labels-hf  (label-only parquet mirror)")
            print(f"[keys]   3. manual download of {KEYS_URL}")
            if os.path.exists(out):
                os.remove(out)  # drop the 596-byte 503 HTML error page
            return None
    got = md5_of_file(out)
    if got == KEYS_MD5:
        print(f"[keys] md5 OK ({KEYS_MD5})")
    else:
        print(f"[keys] WARNING md5 {got} != {KEYS_MD5} (organizers have updated the bundle before)")
    with tarfile.open(out) as tar:
        tar.extractall(dest)
    print(f"[keys] extracted under {dest}/keys/")
    return out


def download_label_parquet_hf(dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, "LA_eval_labels_hf.parquet")
    print(f"[hf-labels] {HF_LABELS_URL}")
    http_download_small(HF_LABELS_URL, out)
    print(f"[hf-labels] saved {out} (utterance_id + label columns, 1=bonafide, 0=spoof)")
    return out


def download_trl_list(dest: str) -> str:
    """The bare challenge-time trial list (mirrors what is inside the Zenodo
    tarball). Used by --only-protocols so you don't need the 7.8 GB archive."""
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, "ASVspoof2021.LA.cm.eval.trl.txt")
    if os.path.exists(out) and md5_of_file(out) == TRL_LIST_MD5:
        print(f"[trl] {out} already present, md5 OK")
        return out
    print(f"[trl] {TRL_FALLBACK_URL}")
    http_download_small(TRL_FALLBACK_URL, out)
    got = md5_of_file(out)
    if got != TRL_LIST_MD5:
        print(f"[trl] WARNING md5 {got} != {TRL_LIST_MD5}")
    else:
        print(f"[trl] md5 OK ({TRL_LIST_MD5})")
    return out


# --------------------------------------------------------------------------
# Kaggle fallback
# --------------------------------------------------------------------------
def download_audio_from_kaggle(dest: str) -> None:
    """Full mirror download via the kaggle CLI (~58 GB incl. DF eval part00!)."""
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", dest, "--unzip",
    ]
    print(f"[kaggle] {' '.join(cmd)}")
    print(f"[kaggle] NOTE: the mirror is ~58 GB and also contains DF eval data.")
    print(f"[kaggle] Requires a Kaggle API token (~/.kaggle/kaggle.json).")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify(dest: str) -> int:
    print(f"verifying {dest}")
    problems = 0
    audio_dirs = [
        os.path.join(dest, "ASVspoof2021_LA_eval", "flac"),
        os.path.join(dest, "flac"),
    ]
    audio_dir = next((d for d in audio_dirs if os.path.isdir(d)), None)
    if audio_dir:
        n = sum(1 for f in os.listdir(audio_dir) if f.endswith(".flac"))
        print(f"  flac audio files: {n} (expected {N_EVAL_TRIALS})")
        if n != N_EVAL_TRIALS:
            problems += 1
    else:
        print("  no LA eval flac directory found (audio not downloaded)")
    for rel in [
        "ASVspoof2021.LA.cm.eval.trl.txt",
        os.path.join("ASVspoof2021_LA_eval", "ASVspoof2021.LA.cm.eval.trl.txt"),
    ]:
        p = os.path.join(dest, rel)
        if os.path.exists(p):
            n = sum(1 for _ in open(p))
            ok = "md5 OK" if md5_of_file(p) == TRL_LIST_MD5 else "md5 MISMATCH"
            print(f"  trial list {rel}: {n} lines ({ok})")
            if n != N_EVAL_TRIALS:
                problems += 1
    keys = os.path.join(dest, "keys", "LA", "CM", "trial_metadata.txt")
    if os.path.exists(keys):
        n = sum(1 for _ in open(keys))
        print(f"  keys/LA/CM/trial_metadata.txt: {n} trials WITH labels")
        if n != N_EVAL_TRIALS:
            problems += 1
    else:
        print("  keys/LA/CM/trial_metadata.txt MISSING (eval labels not downloaded)")
        problems += 1
    asv = os.path.join(dest, "keys", "LA", "ASV", "ASVTorch_Kaldi", "score.txt")
    if os.path.exists(asv):
        n = sum(1 for _ in open(asv))
        print(f"  keys/LA/ASV/ASVTorch_Kaldi/score.txt: {n} official ASV scores")
    else:
        print("  official ASV scores MISSING (needed only for recomputing t-DCF C012)")
    lbl = os.path.join(dest, "LA_eval_labels_hf.parquet")
    if os.path.exists(lbl):
        print(f"  {lbl}: present (label-only fallback)")
    print("PROBLEMS" if problems else "all good")
    return 1 if problems else 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join("data", "raw", "asvspoof2021"),
                    help="download directory (default: data/raw/asvspoof2021)")
    ap.add_argument("--record_id", default=DEFAULT_RECORD_ID,
                    help=f"Zenodo record id for the LA eval set (default: {DEFAULT_RECORD_ID})")
    ap.add_argument("--only-protocols", action="store_true",
                    help="skip the 7.8 GB audio; fetch the trial list, eval keys/meta-labels "
                         "and (inside the keys bundle) the official ASV scores")
    ap.add_argument("--source", choices=["zenodo", "kaggle"], default="zenodo",
                    help="where to get the audio (default: zenodo)")
    ap.add_argument("--no-extract", action="store_true", help="do not untar archives")
    ap.add_argument("--no-keys", action="store_true",
                    help="skip the asvspoof.org keys bundle (labels/ASV scores)")
    ap.add_argument("--key-retries", type=int, default=20,
                    help="how hard to hammer asvspoof.org for the keys bundle "
                         "before giving up (server answers 503 when busy)")
    ap.add_argument("--labels-hf", action="store_true",
                    help="also download the label-only parquet from the HuggingFace mirror")
    ap.add_argument("--verify", action="store_true",
                    help="print what is already on disk (audio count, protocol lines, keys) and exit")
    args = ap.parse_args()

    if args.verify:
        return verify(args.dest)

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    if args.only_protocols:
        download_trl_list(dest)
        if not args.no_keys:
            download_keys(dest, retries=args.key_retries)
        if args.labels_hf:
            download_label_parquet_hf(dest)
        return verify(dest)

    if args.source == "zenodo":
        download_audio_from_zenodo(dest, args.record_id, extract=not args.no_extract)
    else:
        download_audio_from_kaggle(dest)

    # The audio tarball contains only the bare trial list; labels still come
    # from the keys bundle.
    if not args.no_keys:
        download_keys(dest, retries=args.key_retries)
    if args.labels_hf:
        download_label_parquet_hf(dest)
    return verify(dest)


if __name__ == "__main__":
    sys.exit(main())
