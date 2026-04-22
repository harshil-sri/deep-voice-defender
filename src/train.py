import os
import sys

# Auto-configure LD_LIBRARY_PATH for NVIDIA pip packages so TensorFlow detects the GPU.
if "LD_LIBRARY_PATH_CONFIGURED" not in os.environ:
    try:
        import nvidia
        nvidia_base = nvidia.__path__[0]
        libs = [os.path.join(nvidia_base, lib, "lib") for lib in os.listdir(nvidia_base) if os.path.isdir(os.path.join(nvidia_base, lib, "lib"))]
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs) + (":" + current_ld if current_ld else "")
        
        # Configure XLA to find libdevice and ptxas
        cuda_nvcc_path = os.path.join(nvidia_base, "cuda_nvcc")
        if os.path.exists(cuda_nvcc_path):
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_nvcc_path}"
            # Add ptxas to PATH
            os.environ["PATH"] = os.path.join(cuda_nvcc_path, "bin") + ":" + os.environ.get("PATH", "")
            
        os.environ["LD_LIBRARY_PATH_CONFIGURED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"Warning: Auto-config failed: {e}")
        pass

import numpy as np
import math as m
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from preprocessing import load_audio,audio_to_mel
from augmentation import augment_audio
from model import build_model
import librosa
def get_labels(protocol_file_path):
    labels_dict={}
    with open(protocol_file_path, 'r') as file:
        for line in file:
            parts=line.strip().split()
            #file id
            file_id=parts[1]
            #last word
            label_text=parts[-1]
            
            #convert to numbers
            if label_text=="bonafide":
                labels_dict[file_id]=1
            else:
                labels_dict[file_id]=0
    return labels_dict
            
class AudioDataGenerator(Sequence):
    def __init__(self,file_id,labels_dict,audio_dir,batch_size=32):
        self.file_id=file_id
        self.labels_dict=labels_dict
        self.audio_dir=audio_dir
        self.batch_size=batch_size
        self.on_epoch_end() # Shuffle before the very first epoch
        
        
    def __len__(self):
        return m.ceil(len(self.file_id)/self.batch_size)
    
    def on_epoch_end(self):
        # SHUFFLE THE DATA! The ASVspoof protocol file is sorted by class. 
        # Without shuffling, the model sees all 'Bonafide' first, then all 'Spoof',
        # causing catastrophic forgetting where it just guesses 'Spoof' for everything!
        np.random.shuffle(self.file_id)
    
    
    def __getitem__(self,index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_ids= self.file_id[start_index : end_index]
        
        x=[]
        y=[]
        
        for file_id in batch_ids:
            #Build Path
            file_path=os.path.join(self.audio_dir, file_id + ".flac")
            
            #Load Audio
            raw_audio=load_audio(file_path)
            
            #Augment Audio
            aug_audio=augment_audio(raw_audio)
            
            #Spectrogram Making
            spectrogram=audio_to_mel(aug_audio)
            
            #Get Answer
            label=self.labels_dict[file_id]
            x.append(spectrogram)
            y.append(label)
        return np.expand_dims(np.array(x), axis=-1), np.array(y)
    
if __name__ == "__main__":
    #Define where your data is
    protocol_path = "data/raw/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = "data/raw/LA/LA/ASVspoof2019_LA_train/flac"
    
    print("Loading Answer Key...")
    labels_dict = get_labels(protocol_path)
    
    #Get a list of just the file names (e.g., ['LA_T_1138215', 'LA_T_1234567', ...])
    file_ids = list(labels_dict.keys())
    print(f"Found {len(file_ids)} training files!")
    
    #Start the Conveyor Belt
    print("Starting the Data Generator...")
    train_generator = AudioDataGenerator(file_ids, labels_dict, audio_dir, batch_size=32)
    
    #Build the Brain
    print("Building the Model...")
    # (128, 128, 1) is the shape of your spectrograms
    model = build_model((128, 128, 1)) 
    
    #Compile the Brain
    # Using a lower learning rate (0.0002) to prevent the model from diverging
    # Added AUC metric to better track performance on imbalanced data
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # --- FIX: Class Weights to combat Imbalance ---
    # With shuffling fixed, we can safely apply the true 9:1 mathematical weight 
    # to perfectly balance Bonafide vs Spoof without the model exploding.
    class_weight_dict = {0: 1.0, 1: 9.0}

    # --- FIX: Callbacks (Early Stopping & LR Scheduler) ---
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    #START TRAINING!
    print("🚀 IGNITION: Starting Training Loop...")
    #We will do 40 epochs with early stopping
    model.fit(
        train_generator, 
        epochs=40, 
        class_weight=class_weight_dict,
        callbacks=[early_stopping, lr_scheduler]
    ) 
    
    #Save the trained brain to your hard drive
    print("Training complete. Saving model...")
    model.save("deep_voice_defender_v1.keras")
    print("Model saved successfully!")         