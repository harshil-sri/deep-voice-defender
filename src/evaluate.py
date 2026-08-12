import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__)))
from preprocessing import load_audio, audio_to_mel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        eer = 0.5
    return eer, fpr, tpr, thresholds

def compute_min_tDCF(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    c_miss = 1.0
    c_fa = 1.0
    p_target = 0.01
    t_dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    return np.min(t_dcf)

def evaluate_model(model_path, protocol_file, audio_dir, output_dir="results"):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_path}")
    print(f"PROTOCOL: {protocol_file}")
    print(f"DATA DIR: {audio_dir}")
    print(f"{'='*60}\n")

    model = tf.keras.models.load_model(model_path, compile=False)

    labels = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label_text = parts[-1]
            labels[file_id] = 1 if label_text == "bonafide" else 0

    print(f"Loaded {len(labels)} entries from protocol")

    y_true = []
    y_scores = []
    y_pred = []
    file_ids = []
    latencies = []

    for file_id, label in labels.items():
        flac_path = os.path.join(audio_dir, f"{file_id}.flac")
        if not os.path.exists(flac_path):
            print(f"Missing: {flac_path}")
            continue

        start = tf.timestamp()
        raw_audio = load_audio(flac_path)
        spectrogram = audio_to_mel(raw_audio)
        input_data = np.expand_dims(np.expand_dims(spectrogram, 0), -1)

        prediction = model.predict(input_data, verbose=0)
        score = float(prediction[0][0])

        end = tf.timestamp()
        latency_ms = (end - start).numpy() * 1000

        y_true.append(label)
        y_scores.append(score)
        y_pred.append(1 if score >= 0.5 else 0)
        file_ids.append(file_id)
        latencies.append(latency_ms)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)

    print(f"Evaluated {len(y_true)} files")

    eer, fpr, tpr, thresholds = compute_eer(y_true, y_scores)
    min_tdcf = compute_min_tDCF(y_true, y_scores)
    acc = np.mean(y_pred == y_true)
    auc_score = auc(fpr, tpr)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"EER:           {eer*100:.2f}%")
    print(f"min t-DCF:     {min_tdcf:.4f}")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"AUC:           {auc_score:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1:            {f1:.4f}")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"Avg Latency:   {avg_latency:.1f} ms")
    print(f"P50 Latency:   {p50_latency:.1f} ms")
    print(f"P95 Latency:   {p95_latency:.1f} ms")

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "model_path": model_path,
        "protocol_file": protocol_file,
        "audio_dir": audio_dir,
        "timestamp": datetime.now().isoformat(),
        "num_samples": int(len(y_true)),
        "metrics": {
            "eer": float(eer),
            "eer_percent": float(eer * 100),
            "min_tDCF": float(min_tdcf),
            "accuracy": float(acc),
            "auc": float(auc_score),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "latency_ms": {
                "mean": float(avg_latency),
                "p50": float(p50_latency),
                "p95": float(p95_latency)
            }
        }
    }

    model_name = os.path.basename(model_path).replace('.keras', '')
    json_path = os.path.join(output_dir, f"eval_{model_name}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, f"roc_{model_name}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    fnr = 1 - tpr
    plt.plot(fpr, fnr, label='DET Curve', linewidth=2, color='orange')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title(f'DET Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    det_path = os.path.join(fig_dir, f"det_{model_name}.png")
    plt.savefig(det_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    classes = ['Spoof', 'Bonafide']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, f"cm_{model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Saved figures to {fig_dir}/")

    return results

if __name__ == "__main__":
    MODEL_PATH = "deep_voice_defender_v1.keras"
    PROTOCOL_FILE = "data/raw/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    AUDIO_DIR = "data/raw/LA/LA/ASVspoof2019_LA_eval/flac"
    OUTPUT_DIR = "results"

    if not os.path.exists(MODEL_PATH):
