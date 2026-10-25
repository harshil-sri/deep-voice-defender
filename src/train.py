import os
import sys
import argparse
import math as m
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from preprocessing import load_audio, audio_to_mel
from augmentation import augment_audio
from model import build_model

def get_labels(protocol_file_path):
    labels_dict = {}
    with open(protocol_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_id = parts[1]
            label_text = parts[-1]
            labels_dict[file_id] = 1 if label_text == "bonafide" else 0
    return labels_dict

class AudioDataGenerator(Sequence):
    def __init__(self, file_ids, labels_dict, audio_dir, batch_size=32, aug_mode="full", seed=42, shuffle=True):
        self.file_ids = file_ids
        self.labels_dict = labels_dict
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.seed = seed
        self.shuffle = shuffle
        np.random.seed(seed)
        self.on_epoch_end()

    def __len__(self):
        return m.ceil(len(self.file_ids) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_ids)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.file_ids))
        batch_ids = self.file_ids[start_index:end_index]

        x = []
        y = []

        for file_id in batch_ids:
            file_path = os.path.join(self.audio_dir, file_id + ".flac")
            raw_audio = load_audio(file_path)
            aug_audio = augment_audio(raw_audio, mode=self.aug_mode)
            spectrogram = audio_to_mel(aug_audio)
            label = self.labels_dict[file_id]
            x.append(spectrogram)
            y.append(label)
        return np.expand_dims(np.array(x), axis=-1), np.array(y)

def train_model(aug_mode="full", seed=42, output_name=None, epochs=100, use_class_weights=False):
    train_protocol = "data/raw/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = "data/raw/LA/LA/ASVspoof2019_LA_train/flac"

    print(f"\n{'='*60}")
    print(f"TRAINING: aug_mode={aug_mode}, seed={seed}, epochs={epochs}")
    print(f"{'='*60}\n")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("Loading Answer Key...")
    train_labels = get_labels(train_protocol)
    train_ids = list(train_labels.keys())
    print(f"Train: {len(train_ids)} files")

    print("Starting Data Generator...")
    train_generator = AudioDataGenerator(train_ids, train_labels, audio_dir, batch_size=32, aug_mode=aug_mode, seed=seed, shuffle=True)

    print("Building the Model...")
    model = build_model((128, 128, 1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    class_weight_dict = {0: 1.0, 1: 9.0} if use_class_weights else None
    if use_class_weights:
        print("Using class weights: {0: 1.0, 1: 9.0}")
    else:
        print("No class weights (matching original training)")

    # No validation - ASVspoof dev has unknown attacks, causes overfitting
    # Early stopping on training loss with higher patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=15, restore_best_weights=True, verbose=1)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)

    log_dir = f"logs/fit/{aug_mode}_seed{seed}"
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print(f"🚀 IGNITION: Starting Training Loop (aug_mode={aug_mode}, seed={seed})...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, lr_scheduler, tensorboard_cb]
    )

    if output_name is None:
        output_name = f"deep_voice_defender_{aug_mode}_seed{seed}.keras"

    print(f"Training complete. Saving model to {output_name}...")
    model.save(output_name)
    print("Model saved successfully!")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep-Voice Defender with specified augmentation mode")
    parser.add_argument("--aug_mode", type=str, default="full",
                        choices=["none", "opus", "noise", "full"],
                        help="Augmentation mode: none, opus, noise, full")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model filename")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum epochs")
    parser.add_argument("--class_weights", action="store_true",
                        help="Use class weights (default: False to match original)")
    args = parser.parse_args()

    train_model(aug_mode=args.aug_mode, seed=args.seed, output_name=args.output,
                epochs=args.epochs, use_class_weights=args.class_weights)