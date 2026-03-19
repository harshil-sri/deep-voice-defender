import sys
import os
import numpy as np
import tensorflow as tf
from preprocessing import load_audio, audio_to_mel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def predict_suspect(file_path):
    print("\n🔍 Loading Deep-Voice Defender...")
    try:
        model = tf.keras.models.load_model("deep_voice_defender_v1.keras")
    except Exception as e:
        print("❌ Error: Could not find 'deep_voice_defender_v1.keras'. Did you run train.py?")
        return

    print(f"🎧 Analyzing suspect file: {file_path}")
    
    raw_audio = load_audio(file_path)
    spectrogram = audio_to_mel(raw_audio)
    
    input_data = np.expand_dims(spectrogram, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    
    prediction = model.predict(input_data, verbose=0)
    score = prediction[0][0] 
    
    print("\n" + "="*40)
    print(" 🛡️ DEEP-VOICE DEFENDER VERDICT 🛡️")
    print("="*40)
    
    if score >= 0.5:
        print(f"  🟢 Result: BONAFIDE (REAL HUMAN)")
        print(f"  📊 Confidence: {score * 100:.2f}%")
    else:
        print(f"  🔴 Result: SPOOF (AI DEEPFAKE)")
        print(f"  📊 Confidence: {(1 - score) * 100:.2f}%")
    print("="*40 + "\n")

# The __name__ check ensures this block ONLY runs if you type 'python predict.py' in the terminal.
# sys.argv is a list of the words you typed in the terminal command.
# sys.argv[0] is the script name ("src/predict.py").
# sys.argv[1] is the target audio file path you passed in. If it's missing, the script warns you.
target_file = "data/raw/LA/LA/ASVspoof2019_LA_eval/flac/LA_E_2993349.flac"
if __name__ == "__main__":
    predict_suspect(target_file)