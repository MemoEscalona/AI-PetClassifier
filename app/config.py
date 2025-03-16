import os

# Rutas de los datos
RUTA_ZIP = "archive.zip"
RUTA_EXTRACCION = "extracted"
RUTA_DATOS = "dataset/train"
RUTA_MODELO = "models/cats_vs_dogs.h5"

# Rutas de archivos
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Obtiene el directorio base
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../static/uploads")  # Ruta donde se guardarán las imágenes subidas

# Parámetros de imágenes
TAMAÑO_IMAGEN = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50