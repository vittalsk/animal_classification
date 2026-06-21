from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model

model = tf.keras.models.load_model(
    "image_classifier.keras"
)

# Classes

classes = [
'antelope',
'bear',
'beaver',
'bee',
'bison',
'blackbird',
'buffalo',
'butterfly',
'camel',
'cat',
'cheetah',
'chimpanzee',
'chinchilla',
'cow',
'crab',
'crocodile',
'deer',
'dog',
'dolphin',
'donkey',
'duck',
'eagle',
'elephant',
'falcon',
'ferret',
'flamingo',
'fox',
'frog',
'giraffe',
'goat',
'goose',
'gorilla',
'grasshopper',
'hawk',
'hedgehog',
'hippopotamus',
'hyena',
'iguana',
'jaguar',
'kangaroo',
'koala',
'lemur',
'leopard',
'lizard',
'lynx',
'mole',
'mongoose',
'ostrich',
'otter',
'owl',
'panda',
'peacock',
'penguin',
'porcupine',
'raccoon',
'seal',
'sheep',
'snail',
'snake',
'spider',
'squid',
'unknown',
'walrus',
'whale',
'wolf'
]

IMG_SIZE = 128

# Upload Folder

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            prediction="No file selected"
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            prediction="No file selected"
        )

    # Save Uploaded Image

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    # Read Image

    image = Image.open(
        filepath
    )

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img = np.array(
        image
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # Predict

    pred = model.predict(
        img,
        verbose=0
    )

    idx = np.argmax(pred)
    
    top5 = np.argsort(pred[0])[-5:][::-1]
    print("\nTop 5 Predictions:")
    for i in top5:
    	print(classes[i],f"{pred[0][i] * 100:.2f}%")
    confidence = np.max(pred) * 100
    if confidence < 80:
    	result = "Unknown"
    else:
    	result = classes[idx]
    	
    print("Image Path:", filepath)
    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence:.2f}",
        image_path=f"uploads/{file.filename}"
    )

if __name__ == "__main__":

    app.run(
        debug=True
    )
