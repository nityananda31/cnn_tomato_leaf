
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, 'tomato', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'tomato', 'val')
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20


datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

input_shape = (64, 64, 3)
num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())


def build_cnn1():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_cnn2():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_save_models():
    model1 = build_cnn1()
    model1.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model1.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3)])
    model1.save("model_cnn1.h5")

    model2 = build_cnn2()
    model2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3)])
    model2.save("model_cnn2.h5")


train_and_save_models()
