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
from scipy.spatial import KDTree
import random

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60
ENC_SIZE = 30

MODEL_TYPES = ["-", "PCA", "Deep NN", "Convolutional NN"]
current_model_type = '-'

print("Starting ...")

# Load faces.
def face_from_file(filename):
    return cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255

print('Loading faces...')
faces_dir = 'photos/faces_pure/'
faces_names = os.listdir(faces_dir)
scaled_faces = np.array([ face_from_file(faces_dir + f) for f in faces_names ])


# Class which takes care of a whole model and offers functions to encode and
# decode data.
class CModel:
    def _after_init(self, encoded_faces_noscale):
        # Scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(encoded_faces_noscale)

        # Encode & decode faces
        self.encoded_faces = self.encode_more(scaled_faces)
        self.decoded_faces = self.decode_more(self.encoded_faces)

        # KD Tree
        self.encoding_kd = KDTree(self.encoded_faces)

    def encode(self, image):
        return self.encode_more(image.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)).reshape(ENC_SIZE)

    def encode_more(self, images):
        pass

    def decode(self, encoding):
        return self.decode_more(encoding.reshape(1, ENC_SIZE)).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def decode_more(self, encodings):
        pass

    def closest_match(self, encoding):
        d, i = self.encoding_kd.query(encoding)
        return faces_names[i], self.encoded_faces[i]

    def create_model(model_type):
        return {
            '-': NoModel,
            'PCA': PCAModel,
            'Deep NN': DenseModel,
            'Convolutional NN': ConvoModel
        }[model_type]()


class NoModel(CModel):
    def encode_more(self, images):
        return np.zeros((images.shape[0], ENC_SIZE))

    def decode_more(self, encodings):
        return np.zeros((encodings.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.uint8)

class PCAModel(CModel):
    def __init__(self):
        if not os.path.exists('data/pca.model'):
            pca = sklearn.decomposition.PCA(n_components=ENC_SIZE)
            pca.fit(scaled_faces.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*3))
            with open('data/pca.model', 'wb') as f:
                f.write(pickle.dumps(pca))
        with open('data/pca.model', 'rb') as f:
            self.pca = pickle.loads(f.read())

        self._after_init(self.pca.transform(scaled_faces.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*3)))

    def encode_more(self, images):
        raw = self.scaler.transform(self.pca.transform(images.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*3)))
        return np.clip(raw, -1, 1)

    def decode_more(self, encodings):
        raw = self.pca.inverse_transform(self.scaler.inverse_transform(encodings))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)


class NNModel(CModel):
    def _load_encoder_decoder(self):
        pass

    def __init__(self):
        self.encoder, self.decoder = self._load_encoder_decoder()
        self._after_init(self.encoder.predict(scaled_faces))

    def encode_more(self, images):
        raw = self.scaler.transform(self.encoder.predict(images))
        return np.clip(raw, -1, 1).reshape(-1, ENC_SIZE)

    def decode_more(self, encodings):
        raw = self.decoder.predict(self.scaler.inverse_transform(encodings))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)


class PCANNModel(NNModel):
    def __init__(self):
        self.encoder, self.decoder = self._load_encoder_decoder()
        self.pca = sklearn.decomposition.PCA(n_components=ENC_SIZE)
        enc_faces = self.encoder.predict(scaled_faces)
        self.pca.fit(enc_faces)

        self._after_init(self.pca.transform(enc_faces))

    def encode_more(self, images):
        raw = self.scaler.transform(self.pca.transform(self.encoder.predict(images)))
        return np.clip(raw, -1, 1).reshape(-1, ENC_SIZE)

    def decode_more(self, encodings):
        raw = self.decoder.predict(self.pca.inverse_transform(self.scaler.inverse_transform(encodings)))
        return np.clip(raw*255, 0, 255).astype(np.uint8).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)


class DenseModel(PCANNModel):
    def _load_encoder_decoder(self):
        return keras.models.load_model('data/deep_encoder60.keras'), keras.models.load_model('data/deep_decoder60.keras')


class ConvoModel(PCANNModel):
    def _load_encoder_decoder(self):
        return keras.models.load_model('data/convo_encoder60.keras'), keras.models.load_model('data/convo_decoder60.keras')



print('Loading models...')
models = {x: CModel.create_model(x) for x in MODEL_TYPES}


print('Creating UI...')

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
image_canvas.pack(side=tk.TOP)

# Best Match image.
match_canvas = tk.Canvas(window_frame_mid, width=300, height=300)
match_image = None
match_canvas.pack(side=tk.TOP)


def update_canvas(parameters):
    encoding = np.array(parameters, dtype=np.float64)

    # Update shown image
    global canvas_image
    img = models[current_model_type].decode(encoding)
    img = cv2.resize(img, (300, 300))
    canvas_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    image_canvas.create_image(0, 0, image=canvas_image, anchor=tk.NW)

    # Update closest match
    global match_image
    if current_model_type != '-':
        match_name, match_enc = models[current_model_type].closest_match(encoding)
        img = (face_from_file(faces_dir + match_name) * 255).astype(np.uint8)
        img = cv2.resize(img, (300, 300))
        match_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
        match_canvas.create_image(0, 0, image=match_image, anchor=tk.NW)

        label_name = ' '.join(reversed(match_name.replace('-image.jpg', '').split('-'))).title()
        best_match_label.config(text="Are you {}?".format(label_name))

update_canvas(slider_values)

# Best match text.
best_match_label = tk.Label(window_frame_mid)
best_match_label.pack()

# Search field and button.
search_text_variable = tk.StringVar()
search_text = tk.Entry(window_frame_right, textvariable=search_text_variable)
search_text.pack()

def search():
    raw_text = search_text_variable.get().replace("\n", '').replace(' ', '-').lower() + '-image.jpg'
    filename = faces_dir + raw_text
    img = face_from_file(filename)
    encoding = models[current_model_type].encode(img)
    for x, sl in zip(encoding, sliders_list):
        sl.set(x)

search_button = tk.Button(window_frame_right, text="Search Celebrity", command=search)
search_button.pack()

# Randomizer Button
def random_celeb():
    name = random.choice(faces_names).replace('-image.jpg', '').replace('-', ' ')
    search_text_variable.set(name)
    search()

random_celeb_button = tk.Button(window_frame_right, text="Random Celebrity", command=random_celeb)
random_celeb_button.pack()

print('Ready')

window.mainloop()
