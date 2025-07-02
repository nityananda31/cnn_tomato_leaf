
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# 1. Set Paths
DATASET_PATH = 'tomato'  # Ganti dengan path dataset lokal kamu
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

# 2. Data Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

input_shape = (64, 64, 3)
num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

# 3. Model CNN1
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

# 4. Model CNN2
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

# 5. Train CNN1
model1 = build_cnn1()
model1.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model1.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3)])
model1.save("model_cnn1.h5")

# 6. Train CNN2
model2 = build_cnn2()
model2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3)])
model2.save("model_cnn2.h5")

# 7. Plot Validation Accuracy
def plot_history(histories, labels):
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_accuracy'], label=f'{label} Val Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history([history1, history2], ['CNN1', 'CNN2'])

# 8. Predict Sample from Validation Set
val_data.reset()
for i in range(5):
    img_batch, label_batch = val_data.next()
    img = img_batch[0]
    label = np.argmax(label_batch[0])
    img_input = np.expand_dims(img, axis=0)
    pred1 = model1.predict(img_input, verbose=0)
    pred2 = model2.predict(img_input, verbose=0)
    pred_label1 = np.argmax(pred1)
    pred_label2 = np.argmax(pred2)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {class_names[label]}\nCNN1: {class_names[pred_label1]} | CNN2: {class_names[pred_label2]}")
    plt.show()

# 9. Predict Custom Image File
def predict_custom_image(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    pred1 = model1.predict(img_input, verbose=0)
    pred2 = model2.predict(img_input, verbose=0)
    label1 = class_names[np.argmax(pred1)]
    label2 = class_names[np.argmax(pred2)]
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediksi CNN1: {label1}\nPrediksi CNN2: {label2}")
    plt.show()

# Contoh:
# predict_custom_image("contoh_daun.jpg")
