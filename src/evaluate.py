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

