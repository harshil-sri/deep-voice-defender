# Deep-Voice Defender - Project Context & Roadmap

**Last Updated:** 2026-09-19  
**Status:** Working end-to-end pipeline, zero evaluation, fake metrics in README  
**Target:** Publishable research paper (Option A: 2-week plan)

---

## 1. Current State of the Project

### What Exists (Working Code)

| File | Lines | Description | Verified |
|------|-------|-------------|----------|
| `app.py` | 220 | Streamlit UI — loads model, plots spectrogram, predicts spoof/bonafide | ✅ Runs |
| `src/preprocessing.py` | 40 | `load_audio()` (resample 16kHz, pad/trim to 4s), `audio_to_mel()` (128 mel bins, min-max normalized) | ✅ Runs |
| `src/augmentation.py` | 26 | `simulate_opus()` (16k→8k→16k resample), `add_noise()` (Gaussian, 0.005 factor), `augment_audio()` (50% none, 25% OPUS, 25% noise) | ✅ Runs |
| `src/model.py` | 47 | 3× Conv2D blocks (32→64→128 filters) + BatchNorm + MaxPool + Dropout → Flatten → Dense(128) → Dense(1, sigmoid) | ✅ Runs |
| `src/train.py` | 160 | `AudioDataGenerator` (Keras `Sequence`), class weights {0:1, 1:9}, EarlyStopping + ReduceLROnPlateau, 40 epochs | ✅ Runs |
| `src/predict.py` | 46 | CLI inference — loads `.keras`, runs same preprocessing, prints verdict | ✅ Runs |
| `deep_voice_defender_v1.keras` | ~3MB | Pre-trained model file | ✅ Exists |
| `requirements.txt` | 18 | TensorFlow[CUDA], librosa, streamlit, etc. | ✅ Installs |
| `data/raw/` | — | ASVspoof 2019 LA dataset (train + eval + protocols) | ✅ Downloaded |

### What Does NOT Exist (Critical Gaps)

| Missing Component | Why It Matters |
|-------------------|----------------|
| **Evaluation script** | No way to compute EER, min t-DCF, accuracy, AUC on eval set |
| **Real metrics** | README claims "99% accuracy" and "Works on WhatsApp" — zero evidence |
| **Ablation study** | Cannot isolate contribution of OPUS vs. noise vs. none |
| **Baseline comparison** | No vanilla CNN (no augmentation) to compare against |
| **Training logs/curves** | No TensorBoard, no loss/accuracy plots saved |
| **Cross-corpus test** | Never tested on actual WhatsApp-compressed audio |
| **Statistical rigor** | Single seed, no confidence intervals, no multiple runs |

### The "Fake Numbers" Problem

**README.md currently claims:**
- "99% accuracy on clean data" → **Unverified, no evaluation code exists**
- "Works on WhatsApp" → **Simulates OPUS in training, but never tested on real WhatsApp audio**
- Comparison table (Standard vs. Defender) → **Purely aspirational, no experimental data**

**Graphify analysis confirmed:** 27% of graph edges are INFERRED (not extracted from code). The inferred edges (main→load_audio, predict_suspect→audio_to_mel, etc.) are **actually correct** — the code does call them. But the performance claims are fabricated.

---

## 2. What Option A Is

**Option A = Real Research Paper (2 weeks)**

A complete, honest, reproducible experimental paper with:
1. **Real evaluation** on ASVspoof 2019 LA eval set (EER, min t-DCF, AUC, accuracy)
2. **Ablation study** — 4 training runs: (a) no augmentation, (b) OPUS only, (c) noise only, (d) full
3. **Baseline comparison** — vanilla CNN vs. Defender CNN
4. **Cross-corpus validation** — test on real WhatsApp/Telegram compressed audio
5. **Statistical rigor** — 3 seeds, confidence intervals, training curves
6. **Written paper** — 6-8 pages: Introduction, Related Work, Method, Experiments, Results, Conclusion

**Target venue:** College research journal / conference (BTEC level)  
**Deliverable:** PDF + GitHub repo with reproducible code + trained models

---

## 3. How We Continue (Workflow)

### Phase 1: Evaluation Infrastructure (Days 1-2)
```
Create src/evaluate.py
    ↓
Run on ASVspoof 2019 LA eval set (already in data/raw/)
    ↓
Get: EER, min t-DCF, AUC, accuracy, ROC/DET curves
    ↓
Save results to results/eval_baseline.json
```

### Phase 2: Ablation Training (Days 3-7)
```
Modify train.py to accept --augmentation_mode flag
    ↓
Train 4 variants (3 seeds each = 12 runs):
  1. none      — no augmentation
  2. opus_only — simulate_opus only
  3. noise_only — add_noise only
  4. full      — current augment_audio (50/25/25)
    ↓
Evaluate all 12 models on eval set
    ↓
Statistical comparison (mean ± std EER)
```

### Phase 3: Cross-Corpus Test (Days 8-9)
```
Collect 50+ real WhatsApp voice notes (bonafide + spoof)
    ↓
Create src/test_whatsapp.py
    ↓
Run all 4 model variants on WhatsApp test set
    ↓
Report degradation vs. clean eval
```

### Phase 4: Paper Writing (Days 10-14)
```
Generate all tables/figures from results/
    ↓
Write paper in Overleaf/LaTeX or Markdown→PDF
    ↓
Sections: Abstract, Intro, Related Work, Method, Experiments, Results, Discussion, Conclusion
    ↓
Push final code + models + paper to GitHub
```

---

## 4. Detailed To-Do List
