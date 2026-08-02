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
