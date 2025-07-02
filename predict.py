import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import ImageTk, Image

IMG_SIZE = (64, 64)
MODEL1_PATH = "model_cnn1.h5"
MODEL2_PATH = "model_cnn2.h5"
CLASS_NAMES = sorted(next(os.walk("tomato/train"))[1])


model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)


def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = load_img(file_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred1 = model1.predict(img_input, verbose=0)
    pred2 = model2.predict(img_input, verbose=0)

    idx1 = np.argmax(pred1)
    idx2 = np.argmax(pred2)
    label1 = CLASS_NAMES[idx1]
    label2 = CLASS_NAMES[idx2]
    conf1 = float(np.max(pred1)) * 100
    conf2 = float(np.max(pred2)) * 100

    
    result_label.config(
        text=f"Prediksi CNN1: {label1} ({conf1:.2f}%)\\nPrediksi CNN2: {label2} ({conf2:.2f}%)"
    )
    img_display = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_display)
    image_panel.config(image=img_tk)
    image_panel.image = img_tk


root = tk.Tk()
root.title("Deteksi Daun Tomat - CNN1 vs CNN2")

frame = tk.Frame(root)
frame.pack(pady=10)

btn = tk.Button(frame, text="Pilih Gambar", command=predict_image)
btn.pack()

image_panel = tk.Label(root)
image_panel.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 12), justify="center")
result_label.pack(pady=10)

root.mainloop()
