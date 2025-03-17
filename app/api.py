from flask import Flask,Blueprint, request, jsonify 
import tensorflow as tf
import numpy as np
import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from app.config import RUTA_MODELO, TAMAÑO_IMAGEN

modelo = tf.keras.models.load_model(RUTA_MODELO)
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    file = request.files["file"]
    img = load_img(io.BytesIO(file.read()), target_size=TAMAÑO_IMAGEN)
    file.seek(0)  # Reinicia el puntero del archivo en caso de necesitarlo más adelante
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = modelo.predict(img_array)
    label = "dog" if prediction[0][0] > 0.5 else "cat"

    return jsonify({"prediction": label})