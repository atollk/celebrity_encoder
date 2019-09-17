import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import os
import datetime
import cv2
import PIL
import PIL.ImageTk
import pickle
import keras
import tensorflow as tf

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60
ENC_SIZE = 30

MODEL_TYPES = ["-", "PCA", "Deep NN", "Convolutional NN"]
current_model_type = '-'

print("Starting ...")

# Load faces.
def face_from_file(filename):
    return cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255

faces_dir = 'photos/faces_pure/'
scaled_faces_dict = {f: face_from_file(faces_dir + f) for f in os.listdir(faces_dir)}
scaled_faces = np.array(list(scaled_faces_dict.values()))

# Returns a triple: the loaded model, a data scaler, all encoded faces, and all decoded faces.
def load_model(model_type):
    if model_type == '-':
        return None, scaled_faces, scaled_faces
    elif model_type == 'PCA':
        if not os.path.exists('pca.model'):
            pca = sklearn.decomposition.PCA(n_components=ENC_SIZE)
            pca.fit(scaled_faces.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*3))
            with open('data/pca.model', 'wb') as f:
                f.write(pickle.dumps(pca))
        with open('data/pca.model', 'rb') as f:
            pca = pickle.loads(f.read())
        encoded_faces = pca.transform(scaled_faces.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*3))
        decoded_faces = pca.inverse_transform(encoded_faces).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(encoded_faces)
        return pca, scaler, encoded_faces, decoded_faces
    elif model_type == 'Deep NN':
        encoder = keras.models.load_model('data/deep_encoder60.keras')
        decoder = keras.models.load_model('data/deep_decoder60.keras')
        encoded_faces = encoder.predict(scaled_faces.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        decoded_faces = decoder.predict(encoded_faces).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(encoded_faces)
        return (encoder, decoder), scaler, encoded_faces, decoded_faces
    elif model_type == 'Convolutional NN':
        encoder = keras.models.load_model('data/convo_encoder60.keras')
        decoder = keras.models.load_model('data/convo_decoder60.keras')
        encoded_faces = encoder.predict(scaled_faces.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        decoded_faces = decoder.predict(encoded_faces).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(encoded_faces)
        return (encoder, decoder), scaler, encoded_faces, decoded_faces

def encode_image(image, model_type):
    if model_type == '-':
        return np.zeros(ENC_SIZE, dtype=np.uint8)
    elif model_type == 'PCA':
        raw = pca_scaler.transform(pca.transform(image.reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT * 3)))
    elif model_type == 'Deep NN':
        raw = deep_scaler.transform(deep_encoder.predict(image.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    elif model_type == 'Convolutional NN':
        raw = convo_scaler.transform(convo_encoder.predict(image.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    return np.clip(raw, -1, 1).reshape(ENC_SIZE)

# Returns the decoded image as size IMAGE_WIDTH x IMAGE_HEIGHT in uint8 encoding.
def decode_image(encoding, model_type):
    if model_type == '-':
        return np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.uint8)
    elif model_type == 'PCA':
        raw = pca.inverse_transform(pca_scaler.inverse_transform(encoding))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    elif model_type == 'Deep NN':
        raw = deep_decoder.predict(deep_scaler.inverse_transform(encoding).reshape(-1, ENC_SIZE))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    elif model_type == 'Convolutional NN':
        raw = convo_decoder.predict(convo_scaler.inverse_transform(encoding).reshape(-1, ENC_SIZE))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)


pca, pca_scaler, pca_encoded_faces, pca_decoded_faces = load_model('PCA')
deep_nn, deep_scaler, deep_encoded_faces, deep_decoded_faces = load_model('Deep NN')
deep_encoder, deep_decoder = deep_nn
convo_nn, convo_scaler, convo_encoded_faces, convo_decoded_faces = load_model('Convolutional NN')
convo_encoder, convo_decoder = convo_nn


# Create window.
window = tk.Tk()
window.title("PCA Celebrities")

# We have three frames in our window.
window_frame_left = tk.Frame(window)
window_frame_left.pack(side=tk.LEFT)
window_frame_mid = tk.Frame(window)
window_frame_mid.pack(side=tk.LEFT)
window_frame_right = tk.Frame(window)
window_frame_right.pack(side=tk.LEFT)

# Sliders to control the PCA parameters.
slider_values = [0] * ENC_SIZE

sliders_list = []
for r in range(0, ENC_SIZE//2):
    for c in [0, 1]:
        def slider_callback(new_v, i=2*r+c):
            slider_values[i] = float(new_v)
            update_canvas(slider_values)
        slider = tk.Scale(
            window_frame_left,
            from_=-1,
            to=1,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=slider_callback
        )
        slider.grid(column=c, row=r)
        sliders_list.append(slider)

# Reset button
def reset_sliders():
    for slider in sliders_list:
        slider.set(0)

reset_button = tk.Button(window_frame_left, text="Reset", command=reset_sliders)
reset_button.grid(column=0, row=ENC_SIZE//2)

# Combobox to choose the model.
def switch_model(model_type):
    global current_model_type
    current_model_type = model_type
    update_canvas(slider_values)

model_combobox = ttk.Combobox(window_frame_mid, values=MODEL_TYPES, state='readonly')
model_combobox.current(0)
model_combobox.bind("<<ComboboxSelected>>", lambda ev: switch_model(model_combobox.get()))
model_combobox.pack(side=tk.TOP)


# Combined image.
image_canvas = tk.Canvas(window_frame_mid, width=300, height=300)
canvas_image = None

def update_canvas(parameters):
    global canvas_image
    img = decode_image(np.array(parameters, dtype=np.float64), current_model_type)
    img = cv2.resize(img, (300, 300))
    canvas_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    image_canvas.create_image(0, 0, image=canvas_image, anchor=tk.NW)

update_canvas(slider_values)
image_canvas.pack(side=tk.TOP)

# Search field and button.
search_text = tk.Text(window_frame_right, height=1, width=50)
search_text.pack()

def search():
    raw_text = search_text.get(0.0, tk.END).replace("\n", '').replace(' ', '-').lower() + '-image.jpg'
    filename = faces_dir + raw_text
    img = face_from_file(filename)
    encoding = encode_image(img, current_model_type)
    for x, sl in zip(encoding, sliders_list):
        sl.set(x)

search_button = tk.Button(window_frame_right, text="Search Celebrity", command=search)
search_button.pack()

print('Ready')

window.mainloop()
