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
