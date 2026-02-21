import numpy as np
import librosa
import preprocessing as pp
def simulate_opus(audio_array):
    resampled_array = librosa.resample(audio_array, orig_sr = 16000.00, target_sr = 8000.00)
    opus_array = librosa.resample(resampled_array, orig_sr = 8000.00, target_sr = 16000.00)
    return opus_array

def add_noise(audio):
    x = len(audio)
    random_array = np.random.randn(x)
    noise_factor = 0.005
    random_array *= noise_factor
    noisy_array = audio + random_array
    return(noisy_array)

def augment_audio(audio):
    probability = np.random.random()
    if (probability < 0.5):
        return audio
    elif (0.5 <= probability < 0.75):
        muffled_audio = simulate_opus(audio)
        return muffled_audio
    elif (probability >= 0.75):
        noisy_audio = add_noise(audio)
        return noisy_audio