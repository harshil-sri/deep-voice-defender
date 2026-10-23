"""Protocol file parsing for ASVspoof 2019 LA.

Label convention used everywhere in this repo: 1 = bonafide, 0 = spoof.
Scores: higher = more bonafide (sigmoid output).
"""

import os
import numpy as np

LA_ROOT = "data/raw/LA/LA"
CM_PROTOCOLS = {
    "train": os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"),
}
CM_AUDIO_DIRS = {
    "train": os.path.join(LA_ROOT, "ASVspoof2019_LA_train", "flac"),
    "dev": os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac"),
    "eval": os.path.join(LA_ROOT, "ASVspoof2019_LA_eval", "flac"),
}
ASV_SCORES = {
    "dev": os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_scores", "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"),
    "eval": os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_scores", "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"),
}
ASV_PROTOCOLS = {
    "dev": os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.dev.gi.trl.txt"),
    "eval": os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.eval.gi.trl.txt"),
}


def load_cm_protocol(path):
    """Parse a CM protocol file.

    Line format: `SPEAKER UTTPREFIX_UTTID - ATTACK LABEL`
    Returns (file_ids, labels, attacks): lists of str / int / str.
    `file_ids` are bare utterance ids (e.g. LA_E_2834763), matching the
    flac basenames.
    """
    file_ids, labels, attacks = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt_id = parts[1]
            attack = parts[3]
            label_text = parts[-1]
            file_ids.append(utt_id)
            labels.append(1 if label_text == "bonafide" else 0)
            attacks.append(attack)
    return file_ids, np.array(labels, dtype=np.int64), attacks


def load_asv_trials(score_path, protocol_path):
    """Load ASV scores split by trial key, for official min t-DCF.

    The ASV score file is line-aligned with the ASV protocol file:
      protocol: `SPEAKER UTID ATTACK LABEL`
      scores:   `ATTACK_OR_BONAFIDE KEY SCORE`   (KEY in {target, nontarget, spoof})

    Returns dict with keys tar_asv / non_asv / spoof_asv (np.float64 arrays),
    plus `utt_asv` mapping utterance id -> ASV score.
    """
    utt_asv = {}
    tar, non, spoof = [], [], []
    with open(protocol_path) as fp, open(score_path) as fs:
        for proto_line, score_line in zip(fp, fs):
            proto_parts = proto_line.strip().split()
            score_parts = score_line.strip().split()
            utt_id, key = proto_parts[1], score_parts[1]
            score = float(score_parts[2])
            utt_asv[utt_id] = score
            if key == "target":
                tar.append(score)
            elif key == "nontarget":
                non.append(score)
            else:
                spoof.append(score)
    return {
        "tar_asv": np.array(tar, dtype=np.float64),
        "non_asv": np.array(non, dtype=np.float64),
        "spoof_asv": np.array(spoof, dtype=np.float64),
        "utt_asv": utt_asv,
    }
