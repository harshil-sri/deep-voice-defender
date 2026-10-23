"""Protocol file parsing for ASVspoof 2021 LA.

Label convention used everywhere in this repo: 1 = bonafide, 0 = spoof.
Scores: higher = more bonafide (sigmoid output).

Two DIFFERENT 2021 files exist and both are handled (auto-detected):

1. Challenge-time trial list, `ASVspoof2021.LA.cm.eval.trl.txt`
   (shipped inside the Zenodo audio tarball / Kaggle mirror):
       LA_E_9332881
   ONE column: the utterance id. No speaker, no attack, no label
   (verified against the real file, 181,566 lines,
   md5 2dfda95bec4facf6e4bde655dcffd8a7).

2. Post-challenge keys file, `keys/LA/CM/trial_metadata.txt`
   (from asvspoof.org LA-keys-full.tar.gz), EIGHT space-separated columns:
       spk      trial         codec  trans   attack    label     trim    subset
       LA_0009  LA_E_9332881  alaw   ita_tx  A07       spoof     notrim  eval
       LA_0020  LA_E_7294490  g722   loc_tx  bonafide  bonafide  notrim  eval
   Column names follow the organizers' eval-package config.py
   (asvspoof-challenge/2021, eval-package/config.py):
   ['spk', 'trial', 'codec', 'trans', 'attack', 'label', 'trim', 'subset']
   - `codec` in {none, alaw, pstn, g722, ulaw, gsm, opus} (the 2021
     telephony/channel conditions C1..C7),
   - `trans` is the transducer/transmission condition (e.g. ita_tx, loc_tx),
   - bonafide trials have `bonafide` in BOTH the attack and label columns,
   - spoof trials have the attack id (A07..A19) and label `spoof`.
   This differs from ASVspoof 2019's 5-column
   `SPEAKER UTID - ATTACK LABEL`: 2021 adds codec+trans and (post-challenge
   keys) trim+subset, and the audio tarball's own trl file has NO columns at all.

Directory layout of the 2021 LA eval set:
    ASVspoof2021_LA_eval/
        flac/LA_E_*.flac                    audio (16 kHz mono)
        ASVspoof2021.LA.cm.eval.trl.txt     bare trial list
    keys/                                   from LA-keys-full.tar.gz
        LA/CM/trial_metadata.txt            CM protocol WITH labels
        LA/ASV/trial_metadata.txt           ASV protocol
        LA/ASV/ASVTorch_Kaldi/score.txt     organizers' official ASV scores
        LA/LA-C012-eval.npy                 pre-computed t-DCF C012 coeffs
"""

from __future__ import annotations

import os
import numpy as np

LA2021_ROOT = "data/raw/asvspoof2021"

# Bare challenge-time trial list (inside the audio archive / Zenodo)
TRL_LIST = os.path.join(LA2021_ROOT, "ASVspoof2021_LA_eval", "ASVspoof2021.LA.cm.eval.trl.txt")
# Post-challenge keys (from asvspoof.org LA-keys-full.tar.gz)
CM_TRIAL_METADATA = os.path.join(LA2021_ROOT, "keys", "LA", "CM", "trial_metadata.txt")
ASV_TRIAL_METADATA = os.path.join(LA2021_ROOT, "keys", "LA", "ASV", "trial_metadata.txt")
ASV_SCORES = os.path.join(LA2021_ROOT, "keys", "LA", "ASV", "ASVTorch_Kaldi", "score.txt")

# Column names of the 8-column post-challenge trial_metadata.txt
TRIAL_METADATA_COLS = ["spk", "trial", "codec", "trans", "attack", "label", "trim", "subset"]

# Verified eval-set statistics
N_EVAL_TRIALS = 181566
N_BONAFIDE = 18452
N_SPOOF = 163114


def load_cm_protocol(path, allow_missing_labels=False):
    """Parse a 2021 LA CM protocol file (auto-detects the two formats).

    Returns (file_ids, labels, attacks): lists of str / np.int64 array / list.
    `file_ids` are bare utterance ids (e.g. LA_E_9332881), matching the flac
    basenames.

    - 8-column keys file: labels are 1 (bonafide) / 0 (spoof); attacks are the
      raw attack column ('bonafide' for genuine trials, 'A07'...'A19' for
      spoofed ones).
    - 1-column bare trial list: there are no labels in this file. By default
      this raises ValueError (use `load_cm_trials` instead); pass
      allow_missing_labels=True to get (file_ids, None, None).
    """
    file_ids, labels, attacks = [], [], []
    fmt = None
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if fmt is None:
                fmt = "metadata" if len(parts) >= 8 else "trial_list"
            if fmt == "metadata":
                if len(parts) != len(TRIAL_METADATA_COLS):
                    raise ValueError(f"{path}: expected {len(TRIAL_METADATA_COLS)} columns, got {len(parts)}: {line!r}")
                _, utt_id, _, _, attack, label_text, _, _ = parts
                file_ids.append(utt_id)
                labels.append(1 if label_text == "bonafide" else 0)
                attacks.append(attack)
            else:
                if len(parts) != 1:
                    raise ValueError(f"{path}: trial-list format expects 1 column, got {len(parts)}: {line!r}")
                file_ids.append(parts[0])
    if fmt == "trial_list" and not allow_missing_labels:
        raise ValueError(
            f"{path}: this is the bare 2021 trial list (no labels). Ground-truth "
            f"eval labels are in keys/LA/CM/trial_metadata.txt from "
            f"https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz "
            f"(see scripts/fetch_asvspoof2021.py), or pass allow_missing_labels=True."
        )
    label_arr = np.array(labels, dtype=np.int64) if labels else None
    attack_list = attacks if attacks else None
    return file_ids, label_arr, attack_list


def load_cm_protocol_with_labels(path):
    """Convenience wrapper: parse the 8-column keys file, assert it has labels.

    Returns (file_ids, labels, attacks, meta) where meta is a dict of per-trial
    condition fields (codec / trans / trim / subset / spk), useful for
    condition-wise (codec, transducer) evaluation.
    """
    file_ids, labels, attacks = [], [], []
    meta = {"spk": [], "codec": [], "trans": [], "trim": [], "subset": []}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != len(TRIAL_METADATA_COLS):
                raise ValueError(f"{path}: expected {len(TRIAL_METADATA_COLS)} columns, got {len(parts)}: {line!r}")
            spk, utt_id, codec, trans, attack, label_text, trim, subset = parts
            file_ids.append(utt_id)
            labels.append(1 if label_text == "bonafide" else 0)
            attacks.append(attack)
            meta["spk"].append(spk)
            meta["codec"].append(codec)
            meta["trans"].append(trans)
            meta["trim"].append(trim)
            meta["subset"].append(subset)
    return file_ids, np.array(labels, dtype=np.int64), attacks, meta


def load_cm_trials(path):
    """Parse the bare (label-free) challenge-time trial list.

    Returns a list of utterance ids, in file order.
    """
    ids = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 1:
                raise ValueError(f"{path}: expected a single utterance id per line, got: {line!r}")
            ids.append(parts[0])
    return ids


def load_cm_data(root=LA2021_ROOT, subset="eval"):
    """Locate protocol + audio for the 2021 LA eval set (mirrors 2019's helper).

    Returns (file_ids, labels, attacks, audio_dir).
    Prefers the labelled keys file (keys/LA/CM/trial_metadata.txt); falls back
    to the bare trial list only if labels are truly unavailable anywhere, in
    which case labels/attacks are None.

    `subset` is kept for API symmetry with src/protocol.py; ASVspoof 2021 LA
    ships only the eval partition (train/dev come from the 2019 set).
    """
    if subset != "eval":
        raise ValueError("ASVspoof 2021 LA only has an eval partition; "
                         "use ASVspoof 2019 for train/dev.")
    audio_dir = os.path.join(root, "ASVspoof2021_LA_eval", "flac")
    if not os.path.isdir(audio_dir):
        # the Kaggle mirror unpacks to ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac
        alt = os.path.join(root, "ASVspoof2021_LA_eval", "ASVspoof2021_LA_eval", "flac")
        if os.path.isdir(alt):
            audio_dir = alt
        else:
            alt = os.path.join(root, "flac")
            audio_dir = alt if os.path.isdir(alt) else audio_dir

    candidates = [
        os.path.join(root, "keys", "LA", "CM", "trial_metadata.txt"),
        os.path.join(root, "ASVspoof2021_LA_eval", "trial_metadata.txt"),
        os.path.join(root, "ASVspoof2021_LA_eval", "ASVspoof2021_LA_eval", "trial_metadata.txt"),
        os.path.join(root, "ASVspoof2021_LA_eval", "ASVspoof2021.LA.cm.eval.trl.txt"),
        os.path.join(root, "ASVspoof2021_LA_eval", "ASVspoof2021_LA_eval", "ASVspoof2021.LA.cm.eval.trl.txt"),
        os.path.join(root, "ASVspoof2021.LA.cm.eval.trl.txt"),
    ]
    protocol_path = next((p for p in candidates if os.path.exists(p)), None)
    if protocol_path is None:
        raise FileNotFoundError(
            f"no 2021 LA protocol found under {root}; run "
            f"scripts/fetch_asvspoof2021.py first"
        )
    if protocol_path.endswith("trial_metadata.txt"):
        file_ids, labels, attacks = load_cm_protocol(protocol_path)
    else:
        # bare list: try the keys file for labels before giving up
        keys = candidates[0]
        if os.path.exists(keys):
            file_ids, labels, attacks = load_cm_protocol(keys)
        else:
            file_ids, labels, attacks = load_cm_protocol(protocol_path, allow_missing_labels=True)
    return file_ids, labels, attacks, audio_dir


def load_asv_scores(score_path=ASV_SCORES):
    """Load the organizers' official LA ASV scores (for min t-DCF C012).

    File: keys/LA/ASV/ASVTorch_Kaldi/score.txt, 3 columns
    `asv_spk trial score` (score lines are line-aligned with the ASV protocol
    keys/LA/ASV/trial_metadata.txt, 8 columns, where the label column marks
    target/nontarget/spoof).

    Returns (asv_spk, trials, scores, keys) with keys = per-trial
    'target'/'nontarget'/'spoof' from the ASV protocol.
    """
    proto_path = os.path.join(os.path.dirname(os.path.dirname(score_path)), "trial_metadata.txt")
    asv_spk, trials, scores = [], [], []
    with open(score_path) as fs:
        for line in fs:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 3:
                spk, trial, score = parts
            else:
                raise ValueError(f"{score_path}: expected 3 columns, got {len(parts)}: {line!r}")
            asv_spk.append(spk)
            trials.append(trial)
            scores.append(float(score))
    scores = np.array(scores, dtype=np.float64)
    keys = None
    if os.path.exists(proto_path):
        keys = []
        with open(proto_path) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) == len(TRIAL_METADATA_COLS):
                    keys.append(parts[5])  # label column: target/nontarget/spoof
    return asv_spk, trials, scores, keys
