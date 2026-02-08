import librosa
import numpy as np
import pandas as pd
sample_rate = 16000
duration = 4
audio_len = sample_rate * duration

def load_audio(file_path):
    audio,sr = librosa.load(file_path, sr=16000)
    length_audio = len(audio)
    missing = audio_len - length_audio
    if missing > 0:
        final_array = np.pad(audio,(0,missing), 'constant')
    elif missing < 0:
        final_array = audio[:audio_len]
    else:
        final_array = audio
    return (final_array)

def audio_to_mel(final_array):
    spectrogram_final = librosa.feature.melspectrogram(y = final_array, sr =16000, n_mels = 128)
    x = librosa.power_to_db(spectrogram_final, ref=np.max)
    return(x)

if __name__ == "__main__":
    test_path = "data/raw/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1000648.flac"
    
    try:
        print(f"Testing on: {test_path}")
        audio = load_audio(test_path)
        print(f"✅ Audio Loaded. Shape: {audio.shape} (Should be (64000,))")
        
        spec = audio_to_mel(audio)
        print(f"✅ Spectrogram Created. Shape: {spec.shape} (Should be (128, X))")
    except Exception as e:
        print(f"❌ Error: {e}")