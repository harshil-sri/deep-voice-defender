# ASVspoof 2021 LA — evaluation data: what lives where

Verified 2026-10-23 (live HTTP probes + inspecting real files). Everything below
was checked against the actual sources, not documentation.

## Sources

| Artifact | Where | Verified |
|---|---|---|
| LA eval **audio** (181,566 flac, 16 kHz) | Zenodo record **4837263**, DOI `10.5281/zenodo.4837263` ("ASVspoof 2021 Challenge - Logical Access Database" v1.1, ODC-BY). Single file `ASVspoof2021_LA_eval.tar.gz`, 7,763,666,128 bytes, **md5 `2abee34d8b0b91159555fc4f016e4562`** | yes — record API + file list |
| Bare **trial list** `ASVspoof2021.LA.cm.eval.trl.txt` | inside the tarball above; also mirrored on GitHub (`TakHemlata/SSL_Anti-spoofing`, `database/ASVspoof_LA_cm_protocols/`) | yes — downloaded; md5 `2dfda95bec4facf6e4bde655dcffd8a7` |
| **Eval labels + meta-labels** (codec/transducer/attack per trial) | **NOT on Zenodo.** Post-challenge keys bundle `https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz` (**md5 `f052cc2ed276745afa3b5198665d3b26`**, URL + md5 from the organizers' official eval package `asvspoof-challenge/2021` on GitHub) → untars to `keys/LA/CM/trial_metadata.txt` (CM protocol WITH labels), `keys/LA/ASV/trial_metadata.txt`, C012 npy files | yes — URL/md5 verified via official repo; **download blocked at the moment**: asvspoof.org answers 503 to everything (2026-10-23) |
| **Official ASV scores** (needed to recompute t-DCF C012 for official min t-DCF) | inside the same keys bundle: `keys/LA/ASV/ASVTorch_Kaldi/score.txt`; pre-computed C012 coeffs are also included (`LA/LA-C012-eval.npy`) | yes — documented + md5'd in the official eval-package README; not yet on disk (server down) |
| Label-only fallback (no audio) | HuggingFace `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA`, file `data/labels.parquet` (utterance_id + label; 181,566 rows = 18,452 bonafide / 163,114 spoof, built from the official keys) | yes — downloaded and inspected |
| Kaggle mirror `mohammedabdeldayem/avsspoof-2021` (62 GB) | Contains the **full LA eval partition** (`ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/`: `flac/*.flac`, `ASVspoof2021.LA.cm.eval.trl.txt` — identical md5 to the tarball's copy —, `README.LA.txt`, `LICENSE.txt`) plus **DF eval part00**. **No PA. No keys/labels, no ASV scores.** The trl file it ships is the same 1-column trial list | yes — probed via Kaggle API; downloaded trl list + sample flac |
| Note: Zenodo **4817532** is *PartialSpoof*, not ASVspoof 2021 | — | yes — record API |

## Protocol formats (the important difference vs 2019)

**There is no speaker/attack/label information in the 2021 trl file.** The
challenge-time `ASVspoof2021.LA.cm.eval.trl.txt` shipped with the audio is a
**single-column list of utterance ids**:

```
LA_E_9332881
LA_E_6866159
...
```

The speaker / channel conditions / attack / label come from the post-challenge
keys file `keys/LA/CM/trial_metadata.txt` — **8 space-separated columns**:

```
spk     trial         codec trans  attack   label    trim   subset
LA_0009 LA_E_9332881  alaw  ita_tx A07      spoof    notrim eval
LA_0020 LA_E_7294490  g722  loc_tx bonafide bonafide notrim eval
```

Column names per the organizers' eval package (`asvspoof-challenge/2021`,
`eval-package/config.py`): `['spk', 'trial', 'codec', 'trans', 'attack',
'label', 'trim', 'subset']`.

- `codec` ∈ {none, alaw, pstn, g722, ulaw, gsm, opus} — the 2021 telephony
  conditions (C1..C7 in the results tables).
- `trans` = transducer/transmission condition (e.g. `ita_tx`, `loc_tx`).
- bonafide trials carry `bonafide` in **both** the attack and label columns;
  spoofed trials carry the attack id (A07–A19) and `spoof`.
- `trim` ∈ {trim, notrim}, `subset` ∈ {eval, progress, hidden}.

Compare ASVspoof **2019**: 5 columns `SPEAKER UTID - ATTACK LABEL` — 2021 adds
codec + trans (+ trim/subset in the keys file), and the audio tarball's own
file has no columns at all. `src/protocol_2021.py` auto-detects both 2021
formats (and still returns the `(file_ids, labels, attacks)` contract of
`src/protocol.py`, with labels 1 = bonafide / 0 = spoof).

## Reproducing the download

```bash
# audio (7.8 GB from Zenodo) + bare trial list + keys bundle (best effort)
python3 scripts/fetch_asvspoof2021.py

# protocols / keys / ASV scores only — no audio
python3 scripts/fetch_asvspoof2021.py --only-protocols

# protocols + the label-only parquet fallback from HuggingFace
# (useful while asvspoof.org is down)
python3 scripts/fetch_asvspoof2021.py --only-protocols --labels-hf

# audio from the Kaggle mirror instead of Zenodo (needs kaggle CLI + token;
# downloads the whole ~62 GB mirror, which also includes DF eval part00)
python3 scripts/fetch_asvspoof2021.py --source kaggle

# sanity-check what is already on disk
python3 scripts/fetch_asvspoof2021.py --verify
```

Downloads are resumable (HTTP Range + `.part` temp files) and md5-verified
where the source publishes a checksum.

## Current status / TODO

- [x] Zenodo audio URL, size, md5 verified.
- [x] Bare trial list downloaded from the GitHub mirror; md5 matches.
- [x] Kaggle mirror contents probed (LA eval complete; no labels).
- [x] Keys-bundle URL + md5 verified from the official eval package.
- [ ] **`LA-keys-full.tar.gz` could not be downloaded today**: asvspoof.org
      returns 503 on every path (the organizers' own `download.sh` documents
      the server being frequently busy and just retries). Until it succeeds,
      eval ground-truth labels and official ASV scores are NOT on disk;
      `--labels-hf` provides the labels in the meantime.
- [ ] Once the keys bundle is downloaded, run
      `python3 scripts/fetch_asvspoof2021.py --verify` — it should report
      181,566 flac files and 181,566 labelled trials.
