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
        
        
    def __len__(self):
        return m.ceil(len(self.file_id)/self.batch_size)
    
    
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
    #'adam' is the smart learning algorithm
    #'binary_crossentropy' is the math formula for Yes/No (1/0) decisions
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    #START TRAINING!
    print("🚀 IGNITION: Starting Training Loop...")
    #We will do 5 epochs
    model.fit(train_generator, epochs=5) 
    
    #Save the trained brain to your hard drive
    print("Training complete. Saving model...")
    model.save("deep_voice_defender_v1.keras")
    print("Model saved successfully!")         