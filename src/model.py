import tensorflow as tf
from keras import layers, models
def build_model(input_shape):

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    # Fully Connected
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# --- This runs only when you test this file directly ---
if __name__ == "__main__":
    # 1. Define a dummy input shape (Height, Width, Color Channels)
    # We use (128, 128, 1) because spectrograms are grayscale images
    dummy_shape = (128, 128, 1)
    
    # 2. Build the model
    cnn_model = build_model(dummy_shape)
    
    # 3. Print the report
    cnn_model.summary()