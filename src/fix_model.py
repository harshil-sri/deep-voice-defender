import os
import sys
import zipfile
import io
import h5py
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.join(os.path.dirname(__file__)))
from model import build_model

def load_weights_from_keras_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        weights_data = z.read('model.weights.h5')
    
    weights_dict = {}
    with h5py.File(io.BytesIO(weights_data), 'r') as f:
        def extract(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights_dict[name] = obj[()]
        f.visititems(extract)
    return weights_dict

def build_weight_map(weights_dict):
    layer_weights = {}
    for key, val in weights_dict.items():
        if key.startswith('layers/') and '/vars/' in key:
            layer_name = key.split('/')[1]
            var_idx = int(key.split('/')[-1])
            if layer_name not in layer_weights:
                layer_weights[layer_name] = {}
            layer_weights[layer_name][var_idx] = val
    return layer_weights

def transfer_weights():
    print("Loading weights from old model...")
    weights_dict = load_weights_from_keras_zip("deep_voice_defender_v1.keras")
    layer_weights = build_weight_map(weights_dict)
    
    print("Building fresh architecture...")
    model = build_model((128, 128, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("Building model (calling on dummy input)...")
    dummy = tf.zeros((1, 128, 128, 1))
    _ = model(dummy)
    
    print("Mapping weights...")
    for layer in model.layers:
        if layer.name in layer_weights:
            w = layer_weights[layer.name]
            weights_list = [w[i] for i in sorted(w.keys())]
            layer.set_weights(weights_list)
            print(f"  ✓ {layer.name}: {len(weights_list)} weight arrays")
        else:
            print(f"  ⚠ {layer.name}: no weights found (new layer?)")
    
    print("Saving fixed model...")
    model.save("deep_voice_defender_v1_fixed.keras")
    print("Done! Saved to deep_voice_defender_v1_fixed.keras")
    return True

if __name__ == "__main__":
    transfer_weights()