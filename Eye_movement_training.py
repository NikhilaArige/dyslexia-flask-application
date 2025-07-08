import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm       
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def show_history_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Data Loading and Preprocessing
data, label = [], []
cw_directory = os.getcwd()
folder = os.path.join(cw_directory, 'eye dataset')

for filename in os.listdir(folder):
    sub_dir = os.path.join(folder, filename)
    for img_name in os.listdir(sub_dir):
        img_dir = os.path.join(sub_dir, img_name)
        print(int(filename), img_dir)
        
        img = cv2.imread(img_dir)
        img = cv2.resize(img, (128, 128))

        if len(img.shape) == 3:
            data.append(img / 255.0)
            label.append(int(filename))

# Convert to NumPy Arrays
data = np.array(data, dtype="float32")
label = np.array(label, dtype="int")

# CNN Model Training
def train_CNN(data, label):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(set(label)), activation='softmax')  # Dynamically adjust output neurons
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.20, random_state=42)

    history = model.fit(np.array(X_train), np.array(Y_train), epochs=20, 
                        validation_data=(np.array(X_test), np.array(Y_test)))

    show_history_graph(history)

    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(Y_test), verbose=2)
    print("Testing Accuracy:", test_acc)
    print("Testing Loss:", test_loss)

    model.save('trained_model_CNN1.h5')
    return model

# Train the CNN Model
model_CNN = train_CNN(data, label)

# Predict using the trained model
Y_CNN = model_CNN.predict(data)
