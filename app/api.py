from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from app.config import RUTA_MODELO, TAMAÑO_IMAGEN

app = Flask(__name__)
modelo = tf.keras.models.load_model(RUTA_MODELO)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files["file"]
    img = load_img(file, target_size=TAMAÑO_IMAGEN)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = modelo.predict(img_array)
    label = "dog" if prediction[0][0] > 0.5 else "cat"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
